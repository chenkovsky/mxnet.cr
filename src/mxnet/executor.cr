module MXNet
  # Symbolic Executor component of MXNet
  class Executor
    private def self.dict(name : Array(String), arrays : Array(NDArray))
      raise MXError.new "Duplicate names detected" unless names.to_set.size == names.size
      Hash.zip name, arrays
    end

    @handle : LibMXNet::ExecutorHandle
    @symbol : Symbol
    @arg_arrays : Array(NDArray)?
    @grad_arrays : Array(NDArray)?
    @aux_arrays : Array(NDArray)?
    @outputs : Array(NDArray)

    @arg_dict : Hash(String, NDArray)?
    @grad_dict : Hash(String, NDArray)?
    @aux_dict : Hash(String, NDArray)?
    alias MXMonitorCallback = String, NDArray -> Void
    @monitor_callback : MXMonitorCallback?

    @ctx : Context?
    @grad_req : (Array(String) | Hash(String, String))?
    @group2ctx : Hash(String, Context)?
    protected def initialize(@handle, @symbol)
      @outputs = get_outputs
    end

    def finalize
      LibMXNet.mx_executor_free(@handle)
    end

    # Return a new executor with the same symbol and shared memory,
    # but different input/output shapes.
    # For runtime reshaping, variable length sequences, etc.
    # The returned executor shares state with the current one,
    # and cannot be used in parallel with it.
    # param partialShaping Whether to allow changing the shape of unspecified arguments.
    # param allowUpSizing Whether to allow allocating new ndarrays that's larger than the original.
    # param kwargs Map of string to Shape.
    #                - new shape for arguments.
    # return
    # executor A new executor that shares memory with this.
    def reshape(kwargs : Hash(String, Shape), partial_shaping : Bool = false, allow_up_sizing : Bool = false)
      arg_shapes, _, aux_shapes = @symbol.infert_shape(kwargs)
      raise MXError.new "Insufficient arguement shapes provided." if arg_shapes.nil?
      new_args = Hash(String, NDArray).new
      new_grads = Hash(String, NDArray).new
      @symbol.arguments do |name, i|
        new_shape = arg_shapes[i]
        arr = @arg_arrays[i]
        d_arr = if @grad_arrays.nil?
                  nil
                else
                  @grad_arrays[i]
                end
        if partial_shaping || kwargs.has_key?(name) || new_shape == arr.shape
          if new_shape.prod > arr.shape.prod
            raise MXError.new ("New shape of arg:#{name} larger than original. " +
                               "First making a big executor and then down sizing it is more efficient than the reverse." +
                               "If you really want to up size, set allow_up_sizing = true to enable allocation of new arrays."
              ) unless allow_up_sizing
            new_args[name] = NDArray.empty(new_shape, arr.context)
            if !d_arr.nil?
              new_grads[name] = NDArray.empty(new_shape, d_arr.context)
            end
          else
            new_args[name] = arr.reshape(new_shape.to_a)
            if !d_arr.nil?
              new_grads[name] = d_arr.reshape(new_shape.to_a)
            end
          end
        else
          raise MXError.new ("Shape of unspecified array arg:#{name} changed." +
                             "This can cause the new executor to not share parameteres" +
                             "with the old one. Please check for error in network." +
                             "If this is intended, set partial_shaping = true to suppress this warning.")
        end
      end
      new_auxs = Hash(String, NDArray).new
      @symbol.auxiliary_states.zip(aux_shapes, @aux_arrays) do |name, new_shape, arr|
        if partial_shaping || new_shape == arr.shape
          if new_shape.prod > arr.shape.prod
            raise MXError.new ("New shape of aux:#{name} larger than original. " +
                               "First making a big executor and then down sizing it " +
                               "is more efficient than the reverse." +
                               "If you really want to up size, set allowUpSizing = true " +
                               "to enable allocation of new arrays.") unless allow_up_sizing
            new_auxs[name] = NDArray.empty(new_shape, arr.context)
          else
            new_auxs[name] = arr.reshape(new_shape.to_a)
          end
        else
          raise MXError.new ("Shape of unspecified array aux:#{name} changed." +
                             "This can cause the new executor to not share parameters " +
                             "with the old one. Please check for error in network." +
                             "If this is intended, set partialShaping = true to suppress this warning.")
        end
      end
      @symbol.bind(@ctx, new_args, new_grads, @grad_req, new_auxs, @group2ctx, self)
    end

    private def get_outputs
      nd_handles = Pointer(NDArrayHandle).null
      size = MXUInt.new 0
      check_call(LibMXNet.mx_executor_outputs(@handle, out size, out nd_handles))
      (0...size).map do |idx|
        NDArray.new nd_handles[idx]
      end
    end

    def forward(is_train : Bool = false, kwargs : Hash(String, NDArray)? = nil)
      if !kwargs.nil?
        kwargs.each do |name, array|
          raise MXError.new "Unknown argument #{name}" unless arg_dict.has_key? name
          arr.copy_to(arg_dict[name])
        end
      end
      check_call LibMXNet.mx_executor_forward(@handle, is_train)
    end

    def backward(out_grads : Array(NDArray) = [] of NDArray)
      nd_array_handles = out_grads.map { |x| x.handle }
      check_call LibMXNet.mx_executor_backward(@handle, nd_array_handles)
    end

    def backward(out_grad : NDArray)
      nd_array_handles = [out_grad.handle]
      check_call LibMXNet.mx_executor_backward(@handle, nd_array_handles)
    end

    def moinitor_callback=(callback : MXMonitorCallback)
      @monitor_callback = callback
      boxed_data = Box.box(callback)
      check_call LibMXNet.mx_executor_set_monitor_callback(@handle,
        ->(s : UInt8*, arr : NDArrayHandle, data : Void*) {
          data_as_callback = Box(typeof(callback)).unbox(data)
          data_as_callback.call(String.new s, NDArray.new arr)
        }, boxed_data)
    end

    def arg_dict
      if @arg_dict.nil?
        @arg_dict = dict @symbol.arguments, @arg_arrays
        @arg_dict.as(Hash(String, NDArray))
      else
        @arg_dict
      end
    end

    def grad_dict
      if @grad_dict.nil?
        @grad_dict = dict @symbol.arguments, @grad_arrays
        @grad_dict.as(Hash(String, NDArray))
      else
        @grad_dict
      end
    end

    def aux_dict
      if @aux_dict.nil?
        @aux_dict = dict @symbol.auxiliary_states, @aux_arrays
        @aux_dict.as(Hash(String, NDArray))
      else
        @aux_dict
      end
    end

    def copy_params_from(arg_params : Hash(String, NDArray), aux_params : Hash(String, NDArray)? = nil, allow_extra_params : Bool = false)
      arg_params.each do |name, array|
        if arg_dict.has_key? name
          array.copy_to arg_dict[name]
        else
          raise MXError.new "Find name #{name} that is not in the arguments" unless allow_extra_params
        end
      end
      if !aux_params.nil?
        aux_params.each do |name, array|
          if aux_dict.has_key? name
            array.copy_to aux_dict[name]
          else
            raise MXError.new "Find name #{name} that is not in the auxiliary states" unless allow_extra_params
          end
        end
      end
    end

    def debug_str
      str = Pointer(UInt8).null
      check_call LibMXNet.mx_executor_print(@handle, out str)
      return String.new str
    end
  end
end
