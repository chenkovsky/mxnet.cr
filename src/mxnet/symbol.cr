require "logger"

module MXNet
  class Symbol
    @handle : LibMXNet::SymbolHandle

    def to_unsafe
      @handle
    end

    abstract class Generator
      abstract def generate(key) : Symbol
    end

    def initialize(@handle)
    end

    def finalize
      LibMXNet.mx_symbol_free(@handle)
    end

    def clone
      MXNet.check_call LibMXNet.mx_symbol_copy(@handle, out handle)
      Symbol.new handle
    end

    def [](idx : Int32) : Symbol
      MXNet.check_call LibMXNet.mx_symbol_get_output(@handle, idx, out handle)
      Symbol.new handle
    end

    def [](name : String) : Symbol
      index = -1
      outputs do |output, i|
        if output == name
          raise MXError.new "There are multiple outputs with name #{name}" unless index == -1
          index == i
        end
      end
      raise MXError.new "Cannot find output that matches name #{name}" unless index >= 0
      return self[index]
    end

    # Get a new grouped symbol whose output contains all the internal outputs of this symbol.
    # @return The internal of the symbol.
    def internals : Symbol
      MXNet.check_call LibMXNet.mx_symbol_get_internals(@handle, out handle)
      return Symbol.new handle
    end

    # List all outputs in the symbol.
    # @return : List of all the outputs.
    def outputs
      MXNet.check_call LibMXNet.mx_symbol_list_outputs(@handle, out size, out arr)
      (0...size).each do |i|
        yield String.new(arr[i]), i
      end
    end

    def outputs
      arr = [] of String
      outputs { |s, i| arr << s }
      arr
    end

    # List all the arguments in the symbol.
    # @return Array of all the arguments.
    def arguments
      MXNet.check_call LibMXNet.mx_symbol_list_arguments(@handle, out size, out arr)
      (0...size).each do |i|
        yield String.new(arr[i]), i
      end
    end

    def arguments
      arr = [] of String
      arguments { |s, i| arr << s }
      arr
    end

    # List all auxiliary states in the symbol.
    # @return The names of the auxiliary states.
    # @note
    # Auxiliary states are special states of symbols that do not corresponds to an argument,
    # and do not have gradient. But still be useful for the specific operations.
    # A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
    # Most operators do not have Auxiliary states.
    def auxiliary_states
      MXNet.check_call LibMXNet.mx_symbol_list_auxiliary_states(@handle, out size, out arr)
      (0...size).each do |i|
        yield String.new(arr[i]), i
      end
    end

    def auxiliary_states
      arr = [] of String
      auxiliary_states { |s, i| arr << s }
      arr
    end

    private def infer_type_impl(keys : UInt8**, sdata : Array(Int32)) : {Array(MXType)?, Array(MXType)?, Array(MXType)?}
      MXNet.check_call LibMXNet.mx_symbol_infer_type(
        @handle, sdata.size,
        keys, sdata,
        out arg_type_size,
        out arg_type_data,
        out out_type_size,
        out out_type_data,
        out aux_type_size,
        out aux_type_data,
        out complete
      )
      if complete != 0
        return {(0...arg_type_size).map { |i| MXType.new arg_type_data[i] },
          (0...out_type_size).map { |i| MXType.new out_type_data[i] },
          (0...aux_type_size).map { |i| MXType.new aux_type_data[i] }}
      else
        return {nil, nil, nil}
      end
    end

    def infer_type(*args)
      keys = Pointer(UInt8*).null
      raise MXError.new "args is not MXType Array" unless args.all? { |arg| arg.is_a? MXType }
      sdata = [] of Int32
      args.each { |x| sdata << x.value }
      infer_shape_impl(keys, sdata)
    end

    def infer_type(**kwargs)
      keys = kwargs.keys.map { |x| x.to_s.to_unsafe }.to_a.to_unsafe
      raise MXError.new "kwargs is not ::Symbol => MXType " unless kwargs.values.all? { |arg| arg.is_a? MXType }
      sdata = [] of Int32
      kwargs.each { |k, v| sdata << v.value }
      infer_shape_impl(keys, sdata)
    end

    def infer_shape(*args)
      infer_shape_impl(false, *args)
    end

    def infer_shape(**kwargs)
      infer_shape_impl(false, **kwargs)
    end

    def infer_shape_partial(*args)
      infer_shape_impl(true, *args)
    end

    def infer_shape_partial(**kwargs)
      infer_shape_impl(true, **kwargs)
    end

    def infer_shape_impl(partial, *args)
      sdata = [] of MXUInt
      ind_ptr = [0_u32]
      keys = nil
      args.each do |s|
        case s
        when Shape
          sdata.concat(s.shape)
        when Array(MXUInt)
          sdata.concat(s)
        when Array(Int32)
          sdata.concat(s.map { |x| x.to_u32 })
        else
          raise MXError.new "args is not array of Shape or Array(MXUInt)"
        end
        ind_ptr << sdata.size.to_u32
      end
      infer_shape_impl(partial, ind_ptr, keys, sdata)
    end

    def infer_shape_impl(partial, **kwargs)
      sdata = [] of MXUInt
      ind_ptr = [0_u32]
      keys = [] of UInt8*
      kwargs.each do |k, v|
        case v
        when Shape
          sdata.concat(v.shape)
        when Array(MXUInt)
          sdata.concat(v)
        when Array(Int32)
          sdata.concat(v.map { |x| x.to_u32 })
        else
          raise MXError.new "kwargs is not namedtuple of Shape or Array(MXUInt)"
        end
        keys << k.to_s.to_unsafe
        ind_ptr << sdata.size.to_u32
      end
      infer_shape_impl(partial, ind_ptr, keys, sdata)
    end

    private def infer_shape_impl(partial : Bool, ind_ptr : Array(MXUInt), sdata : Array(MXUInt)) : {Array(Shape)?, Array(Shape)?, Array(Shape)?}
      if partial
        MXNet.check_call LibMXNet.mx_symbol_infer_shape_partial(
          @handle, ind_ptr.size -1, keys,
          ind_ptr,
          sdata,
          out arg_shape_size,
          out arg_shape_ndim,
          out arg_shape_data,
          out out_shape_size,
          out out_shape_ndim,
          out out_shape_data,
          out aux_shape_size,
          out aux_shape_ndim,
          out aux_shape_data,
          out complete)
      else
        MXNet.check_call LibMXNet.mx_symbol_infer_shape(
          @handle,
          ind_ptr.size -1,
            keys,
            ind_ptr,
            sdata,
            out arg_shape_size,
            out arg_shape_ndim,
            out arg_shape_data,
            out out_shape_size,
            out out_shape_ndim,
            out out_shape_data,
            out aux_shape_size,
            out aux_shape_ndim,
            out aux_shape_data,
            out complete)
      end
      if complete != 0
        return {(0...arg_shape_size).map { |i| Shape.new (0...arg_shape_ndim[i]).map { |j| arg_shape_datap[i][j] } },
          (0...out_shape_size).map { |i| Shape.new (0...out_shape_ndim[i]).map { |j| out_shape_datap[i][j] } },
          (0...aux_shape_size).map { |i| Shape.new (0...aux_shape_ndim[i]).map { |j| aux_shape_datap[i][j] } }}
      else
        return nil, nil, nil
      end
    end

    # Get attribute string from the symbol, this function only works for non-grouped symbol.
    # @param key  The key to get attribute from.
    # @return value The attribute value of the key, returns None if attribute do not exist.
    def attr(key : String) : String?
      MXNet.check_call LibMXNet.mx_symbol_get_attr @handle, key, out ret, out success
      if success != 0
        String.new ret
      else
        nil
      end
    end

    # Invoke symbol as function on inputs.
    # @param name resulting symbol name
    # @param symbols provide named symbols
    # @return the resulting symbol
    def apply(name : String, symbols : Hash(String, Symbol)) : Symbol
      s = clone
      s.compose name, symbols
      s
    end

    # Get a debug string.
    # @return Debug string of the symbol.
    def debug_str
      MXNet.check_call LibMXNet.mx_symbol_print(@handle, out c_str)
      String.new c_str
    end

    # Set the attribute of the symbol.
    def attr=(attr : Hash(String, String))
      attr.each do |k, v|
        MXNet.check_call LibMXNet.mx_symbol_set_attr(@handle, k, v)
      end
    end

    # Save symbol into file.
    # You can also use pickle to do the job if you only work on python.
    # The advantage of load/save is the file is language agnostic.
    # This means the file saved using save can be loaded by other language binding of mxnet.
    # You also get the benefit being able to directly load/save from cloud storage(S3, HDFS)
    #
    # @param fname The name of the file
    #        - s3://my-bucket/path/my-s3-symbol
    #        - hdfs://my-bucket/path/my-hdfs-symbol
    #        - /path-to/my-local-symbol
    # @see Symbol.load : Used to load symbol from file.
    def save(fname : String)
      MXNet.check_call LibMXNet.mx_symbol_save_to_file @handle, fname
    end

    private def symbol_handle(symbols : Array(Symbol))
      return nil, symbols.map { |x| x.to_unsafe }
    end
    private def symbol_handle(symbols : Hash(String, Symbol))
      return symbols.keys.map { |x| x.to_unsafe }, symbols.values.map { |x| x.to_unsafe }
    end
    private def symbol_handle(symbols : Hash(::Symbol, Symbol))
      return symbols.keys.map { |x| x.to_s.to_unsafe }, symbols.values.map { |x| x.to_unsafe }
    end

    private def symbol_handle(**symbols)
      keys = [] of UInt8*
      vals = [] of LibMXNet::SymbolHandle
      symbols.each do |k, v|
        if v.is_a? Symbol
          keys << k.to_s.to_unsafe
          vals << v.to_unsafe
        end
      end
      return keys, vals
    end

    private def symbol_handle(*symbols)
      keys = nil
      vals = [] of LibMXNet::SymbolHandle
      symbols.each do |v|
        if v.is_a? Symbol
          vals << v.to_unsafe
        end
      end
      return keys, vals
    end

    # Compose symbol on inputs.
    # This call mutates the current symbol.
    # @param name resulting symbol name
    # @param symbols provide positional arguments
    # @return the resulting symbol
    def compose(name, symbols : Array(Symbol) | Hash(::Symbol, Symbol))
      keys, args = symbol_handle(symbols)
      MXNet.check_call LibMXNet.mx_symbol_compose(@handle, name, args.size, keys, args)
    end

    def call(*args, **kwargs)
      s = self.clone
      s.compose(*args, **kwargs)
      return s
    end

    protected def compose(*args, **kwargs)
      name = kwargs[:name]?
      if args.size != 0 && kwargs.size != 0
        raise MXError.new "compose only accept input Symbols either as positional or keyword arguments, not both"
      end
      keys, args = if args.size != 0
                     symbol_handle(*args)
                   else
                     symbol_handle(**kwargs)
                   end
      MXNet.check_call LibMXNet.mx_symbol_compose(@handle, name, args.size, keys, args)
    end

    def bind(ctx : Context, grad_req : BindReq, shapes : Hash(String, Shape), types : Hash(String, MXType)? = nil) : Executor
      args = arguments
      types_ = if types.nil?
                 args.map { |a, i| {a, MXType::Float32_T} }.to_h
               else
                 types
               end
      arg_shapes, _, aux_shapes = infer_shape shapes
      arg_types, _, aux_types = infer_type types
      raise MXError.new "Input node is not complete" if arg_shapes.nil? || arg_types.nil?
      arg_ndarrays = arg_shapes.zip(arg_types).map { |s, t|
        # @TODO: NDArray dtype
        NDArray.zeros(s, ctx)
      }
      grad_ndarrays = if grad_req == BindReq::NULL
                        args.zip(arg_shapes, arg_types).map { |name_idx, shape, t|
                          # @TODO: NDArray dtype
                          {name_idx[0], NDArray.zeros(shape, cxt)}
                        }.to_h
                      else
                        nil
                      end
      aux_ndarrays = aux_shapes.zip(aux_types).map { |shape, t| NDArray.zeros shape, ctx }
      bind ctx, arg_ndarrays, grad_ndarrays, grad_req, aux_ndarrays, nil, nil
    end

    def bind(ctx : Context,
             args : Array(NDArray) | Hash(String, NDArray),
             args_grad : Array(NDArray) | Hash(String, NDArray) | Nil,
             grad_req : BindReq | Hash(String, BindReq) | Array(BindReq) = BindReq::Write,
             aux_states : Array(NDArray) | Hash(String, NDArray) | Nil = nil,
             group2ctx : Hash(String, Context)? = nil,
             shared_exec : Executor? = nil) : Executor
      symbol_arguments = arguments
      args_handle, args_ndarray = if args.is_a? Array(NDArray)
                                    raise MXError.new "Length of args do not match number of arguments" unless symbol_arguments.size == args.size
                                    {args.map { |arg| arg.handle }, args}
                                  else
                                    arg_arr = symbol_arguments.map { |arg_name, idx|
                                      raise MXError.new "Must specify all the arguments in args" unless args.has_key? arg_name
                                      args[arg_name]
                                    }
                                    {arg_arr.map { |arg| arg.handle }, arg_arr}
                                  end
      args_grad_handle, args_grad_ndarray = if args_grad.nil?
                                              {Array.new(size: args.size, value: NDArrayHandle.null), nil}
                                            elsif args_grad.is_a? Array(NDArray)
                                              raise MXError.new "Length of args_grad do not match number of arguments" unless symbol_arguments.size == args_grad.size
                                              {args_grad.map { |arg| arg.handle }, args_grad}
                                            else
                                              arg_arr = symbol_arguments.map { |arg_name, idx|
                                                raise MXError.new "Must specify all the arguments in args_grad" unless args_grad.has_key? arg_name
                                                args_grad[arg_name]
                                              }
                                              {arg_arr.map { |arg| arg.handle }, arg_arr}
                                            end
      aux_states_ = auxiliary_states
      aux_args_handle, aux_states_ndarray = if aux_states.nil?
                                              {[] of NDArrayHandle, [] of NDArray}
                                            elsif aux_states.is_a? Array(NDArray)
                                              raise MXError.new "Length of aux_states do not match number of arguments" unless aux_states.size == aux_states_.size
                                              {aux_states.map { |s| s.handle }, aux_states}
                                            else
                                              arg_arr = aux_states_.map { |arg_name, idx|
                                                raise MXError.new "Must specify all the arguments in aux_states" unless aux_states.has_key? arg_name
                                                aux_states[arg_name]
                                              }
                                              {arg_arr.map { |arg| arg.handle }, arg_arr}
                                            end
      grads_req_array = if grad_req.is_a? BindReq
                          Array(BindReq).new(size: symbol_arguments.size, value: grad_req)
                        elsif grad_req.is_a? Array(BindReq)
                          grad_req
                        else
                          symbol_arguments.map do |req|
                            grad_req.fetch(req, BindReq::Null)
                          end
                        end
      reqs_array = grads_req_array.map do |x|
        raise MXError.new "grad_req must be in #{Symbol.bind_req_map}" unless Symbol.bind_req_map.includes? req
        Symbol.bind_req_map x
      end
      ctx_map_keys = [] of UInt8*
      ctx_map_dev_types = [] of Int32
      ctx_map_dev_ids = [] of Int32
      if !group2ctx.nil?
        group2ctx.each do |k, v|
          ctx_map_keys << k.to_unsafe
          ctx_map_dev_types << v.device_type_id
          ctx_map_dev_ids << v.device_id
        end
      end
      exec_handle = ExecutorHandle.null
      shared_handle = if shared_exec.nil?
                        ExecutorHandle.null
                      else
                        shared_exec.handle
                      end
      MXNet.check_call LibMXNet.mx_executor_bind_ex(@handle,
        ctx.device_type_id,
        ctx.device_id,
        ctx_map_keys.size,
        ctx_map_keys,
        ctx_map_dev_types,
        ctx_map_dev_ids,
        args_handle.size,
        args_handle,
        args_grad_handle,
        reqs_array,
        aux_args_handle,
        shared_handle,
        out exec_handle
      )
      executor = Executor.new exec_handle, self.clone
      exeuctor.arg_arrays = args_ndarray
      executor.grad_arrays = args_grad_ndarray
      executor.aux_arrays = aux_states_ndarray
      executor.ctx = Context.new ctx.device_type, ctx.device_id
      executor.grads_req = grads_req
      exeuctor.group2ctx = if group2ctx.nil?
                             nil
                           else
                             group2ctx.map { |k, v| {k, Context.new(v.device_type, v.device_id)} }.to_h
                           end
      executor
    end

    def to_json(io)
      MXNet.check_call LibMXNet.mx_symbol_save_to_json(@handle, out js)
      io << js
    end

    def to_json
      String.build do |io|
        to_json io
      end
    end

    def self.variable(name : String, attr : Hash(String, String)? = nil, shape : Shape | Array(Int32) | Nil = nil) : Symbol
      MXNet.check_call LibMXNet.mx_symbol_create_variable(name, out handle)
      sym = Symbol.new handle
      unless shape.nil?
        attr = {} of String => String if attr.nil?
        attr["__shape__"] = Shape.to_s
      end
      sym.attr = AttrScope[attr]
      sym
    end

    def self.group(symbols : Array(Symbol)) : Symbol
      ihandles = symbols.map &.to_unsafe
      MXNet.check_call LibMXNet.mx_symbol_create_group(ihandles.size, ihandles, out handle)
      Symbol.new handle
    end

    def self.load(fname : String) : Symbol
      MXNet.check_call LibMXNet.mx_symbol_create_from_file fname, out handle
      Symbol.new handle
    end

    def self.from_json(string_or_io : String | IO)
      json = if string_or_io.is_a? IO
               string_or_io.read
             else
               string_or_io
             end
      MXNet.check_call LibMXNet.mx_symbol_create_from_json(json, out handle)
      Symbol.new handle
    end

    protected def self.create(operator : String | Function, *args, **kwargs)
      function = operator.is_a?(String) ? Function[operator] : operator
      param_keys = [] of String
      param_vals = [] of String
      symbol_kwargs = {} of ::Symbol => Symbol
      symbol_args = [] of Symbol
      args.each do |x|
        symbol_args << x if x.is_a? Symbol
      end
      name = kwargs[:name]?
      attr = kwargs[:attr]?
      if function.key_var_num_args && kwargs[function.key_var_num_args.as(String)]?.nil?
        param_keys << function.key_var_num_args.as(String)
        param_vals << args.size.to_s
      end
      kwargs.each do |k, v|
        if k == :name || k == :attr
          next
        end
        if v.is_a? Symbol
          symbol_kwargs[k] = v
        elsif !v.nil?
          param_keys << k.to_s
          param_vals << v.to_s
        end
      end
      c_param_keys = param_keys.map { |s| s.to_unsafe }
      c_param_vals = param_vals.map { |s| s.to_unsafe }
      MXNet.check_call LibMXNet.mx_symbol_create_atomic_symbol(
        function,
        param_keys.size,
        c_param_keys,
        c_param_vals,
        out sym_handle
      )
      if symbol_args.size != 0 && symbol_kwargs.size != 0
        raise MXError.new "#{function.name} can only accept input Symbols either as positional or keyword arguments, not both"
      end
      s = Symbol.new sym_handle
      s.attr = AttrScope[attr]
      hint = function.name.downcase
      name = NameManager[name, hint]
      if symbol_kwargs.size > 0
        s.compose name, symbol_kwargs
      else
        s.compose name, symbol_args
      end
      return s
    end
  end
end

struct Number
  macro def_sym_op(f_name, sym_func)
    private SF_{{sym_func.id}}Scalar = SymbolFunction["{{sym_func.id}}Scalar"]
    def {{f_name.id}}(other : Symbol)
        SF_{{sym_func.id}}Scalar.call(other, scalar: to_s)
    end
  end

  def_sym_op :**, :_RPower
end
