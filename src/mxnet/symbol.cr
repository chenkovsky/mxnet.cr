require "logger"

module MXNet
  class Symbol
    enum BindReq
      NULL  = 0
      WRITE = 1
      ADD   = 3
    end
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

    def dup
      clone
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
          index = i
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
      infer_type_impl(keys, sdata)
    end

    def infer_type(**kwargs)
      keys = kwargs.keys.map { |x| x.to_s.to_unsafe }.to_a.to_unsafe
      raise MXError.new "kwargs is not ::Symbol => MXType " unless kwargs.values.all? { |arg| arg.is_a? MXType }
      sdata = [] of Int32
      kwargs.each { |k, v| sdata << v.value }
      infer_type_impl(keys, sdata)
    end

    def infer_type(args : Array(MXType))
      keys = Pointer(UInt8*).null
      sdata = [] of Int32
      args.each { |x| sdata << x.value }
      infer_type_impl(keys, sdata)
    end

    def infer_type(kwargs : Hash(String, MXType) | Hash(::Symbol, MXType))
      keys = kwargs.keys.map { |x| x.to_s.to_unsafe }.to_a.to_unsafe
      sdata = [] of Int32
      kwargs.each { |k, v| sdata << v.value }
      infer_type_impl(keys, sdata)
    end

    def infer_shape(*args)
      param = infer_shape_param(*args)
      infer_shape_impl *param
    end

    def infer_shape(**kwargs)
      param = infer_shape_param(**kwargs)
      infer_shape_impl *param
    end

    def infer_shape_partial(*args)
      param = infer_shape_param(*args)
      infer_shape_partial_impl *param
    end

    def infer_shape_partial(**kwargs)
      param = infer_shape_param(**kwargs)
      infer_shape_partial_impl *param
    end

    private def infer_shape_param(*args)
      sdata = [] of MXUInt
      ind_ptr = [0_u32]
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
      return ind_ptr, Pointer(UInt8*).null, sdata
    end

    def infer_shape(args : Array(Shape) | Array(Array(Int32)) | Array(Array(MXUInt)))
      param = infer_shape_param(args)
      infer_shape_impl *param
    end

    def infer_shape_partial(args : Array(Shape) | Array(Array(Int32)) | Array(Array(MXUInt)))
      param = infer_shape_param(args)
      infer_shape_partial_impl *param
    end

    def infer_shape(kwargs : Hash(String, Shape) | Hash(String, Array(Int32)) | Hash(::Symbol, Array(MXUInt)) | Hash(::Symbol, Shape) | Hash(::Symbol, Array(Int32)) | Hash(::Symbol, Array(MXUInt)))
      param = infer_shape_param(kwargs)
      infer_shape_impl *param
    end

    def infer_shape_partial(kwargs : Hash(String, Shape) | Hash(String, Array(Int32)) | Hash(::Symbol, Array(MXUInt)) | Hash(::Symbol, Shape) | Hash(::Symbol, Array(Int32)) | Hash(::Symbol, Array(MXUInt)))
      param = infer_shape_param(kwargs)
      infer_shape_partial_impl *param
    end

    private def infer_shape_param(args : Array(Shape) | Array(Array(Int32)) | Array(Array(MXUInt)))
      sdata = [] of MXUInt
      ind_ptr = [0_u32]
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
      return ind_ptr, Pointer(UInt8*).null, sdata
    end

    private def infer_shape_param(**kwargs)
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
      return ind_ptr, keys.to_unsafe, sdata
    end

    private def infer_shape_param(kwargs : Hash(String, Shape) | Hash(String, Array(Int32)) | Hash(::Symbol, Array(MXUInt)) | Hash(::Symbol, Shape) | Hash(::Symbol, Array(Int32)) | Hash(::Symbol, Array(MXUInt)))
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
      return ind_ptr, keys.to_unsafe, sdata
    end

    private def infer_shape_partial_impl(ind_ptr : Array(MXUInt), keys : UInt8**, sdata : Array(MXUInt))
      MXNet.check_call LibMXNet.mx_symbol_infer_shape_partial(
        @handle,
        ind_ptr.size - 1,
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
      if complete != 0
        return {(0...arg_shape_size).map { |i| Shape.new (0...arg_shape_ndim[i]).map { |j| arg_shape_data[i][j] } },
          (0...out_shape_size).map { |i| Shape.new (0...out_shape_ndim[i]).map { |j| out_shape_data[i][j] } },
          (0...aux_shape_size).map { |i| Shape.new (0...aux_shape_ndim[i]).map { |j| aux_shape_data[i][j] } }}
      else
        return nil, nil, nil
      end
    end

    private def infer_shape_impl(ind_ptr : Array(MXUInt), keys : UInt8**, sdata : Array(MXUInt)) : {Array(Shape)?, Array(Shape)?, Array(Shape)?}
      MXNet.check_call LibMXNet.mx_symbol_infer_shape(
        @handle,
        ind_ptr.size - 1,
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
      if complete != 0
        return {(0...arg_shape_size).map { |i| Shape.new (0...arg_shape_ndim[i]).map { |j| arg_shape_data[i][j] } },
          (0...out_shape_size).map { |i| Shape.new (0...out_shape_ndim[i]).map { |j| out_shape_data[i][j] } },
          (0...aux_shape_size).map { |i| Shape.new (0...aux_shape_ndim[i]).map { |j| aux_shape_data[i][j] } }}
      else
        return nil, nil, nil
      end
    end

    def attrs(recursive : Bool = false)
      if recursive
        attrs_recursive
      else
        attrs_shallow
      end
    end

    private def attrs_recursive
      LibMXNet.mx_symbol_list_attr(@handle, out size, out pairs)
      (0...size).map { |i| {String.new(pairs[i << 1]), String.new(pairs[(i << 1) + 1])} }
    end

    private def attrs_shallow
      LibMXNet.mx_symbol_list_attr_shallow(@handle, out size, out pairs)
      (0...size).map { |i| {String.new(pairs[i << 1]), String.new(pairs[(i << 1) + 1])} }
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

    private def compose_param(symbols : Array(Symbol))
      return nil, symbols.map { |x| x.to_unsafe }
    end
    private def compose_param(symbols : Hash(String, Symbol))
      return symbols.keys.map { |x| x.to_unsafe }, symbols.values.map { |x| x.to_unsafe }
    end
    private def compose_param(symbols : Hash(::Symbol, Symbol))
      return symbols.keys.map { |x| x.to_s.to_unsafe }, symbols.values.map { |x| x.to_unsafe }
    end

    private def compose_param(**symbols)
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

    private def compose_param(*symbols)
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
      keys, args = compose_param(symbols)
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
                     compose_param(*args)
                   else
                     compose_param(**kwargs)
                   end
      MXNet.check_call LibMXNet.mx_symbol_compose(@handle, name, args.size, keys, args)
    end

    def simple_bind(ctx : Context,
                    shapes : Hash(String, Shape) | Hash(String, Array(Int32)) | Hash(String, Array(MXUInt)) | Hash(::Symbol, Shape) | Hash(::Symbol, Array(Int32)) | Hash(::Symbol, Array(MXUInt)) = {} of String => Shape,
                    grad_req : BindReq = BindReq::WRITE,
                    types : Hash(String, MXType) | Hash(::Symbol, MXType) | Nil = nil,
                    group2ctx : Hash(String, Context)? = nil) : Executor
      args = arguments
      types_ = types.nil? ? args.map_with_index { |a, i| {a, MXType::Float32_T} }.to_h : types
      arg_shapes, _, aux_shapes = infer_shape shapes
      arg_types, _, aux_types = infer_type types_
      raise MXError.new "Input node is not complete" if arg_shapes.nil? || arg_types.nil?
      raise MXError.new "Input node is not complete" if aux_shapes.nil? || aux_types.nil?
      if group2ctx
        attr_dict = {} of String => Context
        attrs(recursive: true).each do |k, v|
          if k.ends_with? "ctx_group"
            attr_dict[k] = group2ctx.fetch(v, ctx)
          end
        end
        arg_ctx = [] of Context
        aux_ctx = [] of Context
        arguments do |name, idx|
          arg_ctx << attr_dict.fetch(name + "_ctx_group", ctx)
        end
        auxiliary_states do |name, idx|
          aux_ctx << attr_dict.fetch(name + "_ctx_group", ctx)
        end
      else
        arg_ctx = arg_shapes.map { |_| ctx }
        aux_ctx = aux_shapes.map { |_| ctx }
      end
      arg_ndarrays = arg_shapes.zip(arg_types).map { |s, t|
        NDArray.zeros(s, ctx, dtype: t)
      }
      grad_ndarrays = if grad_req == BindReq::NULL
                        (0...args.size).map { |i|
                          name_idx = args[i]
                          shape = arg_shapes[i]
                          actx = arg_ctx[i]
                          t = arg_types[i]
                          {name_idx, NDArray.zeros(shape, actx, dtype: t)}
                        }.to_h
                      else
                        nil
                      end

      aux_ndarrays = (0...aux_shapes.size).map { |i|
        shape = aux_shapes[i]
        actx = aux_ctx[i]
        t = aux_types[i]
        NDArray.zeros(shape, actx, dtype: t)
      }
      bind ctx, arg_ndarrays, grad_ndarrays, grad_req, aux_ndarrays, group2ctx, nil
    end

    private def bind_args_param(args_, args)
      case args
      when Array(NDArray)
        raise MXError.new "Length of args do not match number of arguments" unless args_.size == args.size
        {args.map { |arg| arg.to_unsafe }, args}
      when Hash(String, NDArray)
        arg_arr = [] of NDArray
        args_.each { |arg_name|
          raise MXError.new "Must specify all the arguments in args" unless args.has_key? arg_name
          arg_arr << args[arg_name]
        }
        {arg_arr.map { |arg| arg.to_unsafe }, arg_arr}
      when Hash(::Symbol, NDArray)
        args_ = args.map { |k, v| {k.to_s, v} }.to_h
        bind_args_param(args_, args_)
      when NamedTuple
        hs = Hash(String, NDArray).new
        args.each do |k, v|
          hs[k.to_s] = v
        end
        bind_args_param(args_, hs)
      when Nil
        {[] of LibMXNet::NDArrayHandle, [] of NDArray}
      else
        raise MXError.new "invliad args"
      end
    end

    private def bind_grads_req_param(symbol_arguments, grad_req)
      case grad_req
      when BindReq
        Array(BindReq).new(size: symbol_arguments.size, value: grad_req)
      when Array(BindReq)
        grad_req
      when Hash(String, BindReq)
        ret = [] of BindReq
        symbol_arguments.each do |req|
          ret << grad_req.fetch(req, BindReq::NULL)
        end
        ret
      when Hash(::Symbol, BindReq)
        bind_grads_req_param symbol_arguments, grad_req.map { |k, v| {k.to_s, v} }.to_h
      else
        raise MXError.new "invalid grad_req"
      end
    end

    private def bind_ctx_param(group2ctx)
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
      return ctx_map_keys, ctx_map_dev_types, ctx_map_dev_ids
    end

    def bind(ctx : Context,
             args : Array(NDArray) | Hash(String, NDArray) | Hash(::Symbol, NDArray),
             args_grad : Array(NDArray) | Hash(String, NDArray) | Hash(::Symbol, NDArray) | Nil,
             grad_req : BindReq | Hash(String, BindReq) | Hash(::Symbol, BindReq) | Array(BindReq) = BindReq::WRITE,
             aux_states : Array(NDArray) | Hash(String, NDArray) | Hash(::Symbol, NDArray) | Nil = nil,
             group2ctx : Hash(String, Context)? = nil,
             shared_exec : Executor? = nil) : Executor
      symbol_arguments = arguments
      raise MXError.new "arguments is null" if symbol_arguments.nil?
      args_handle, args_ndarray = bind_args_param(symbol_arguments, args)
      args_grad_handle, args_grad_ndarray = bind_args_param(symbol_arguments, args_grad)
      auxiliary_states_ = auxiliary_states
      raise MXError.new "auxiliary_states is null" if auxiliary_states_.nil?
      aux_args_handle, aux_states_ndarray = bind_args_param(auxiliary_states_, aux_states)
      reqs_array_ = bind_grads_req_param(symbol_arguments, grad_req)
      reqs_array = reqs_array_.map { |x| x.value.to_u32 }
      ctx_map_keys, ctx_map_dev_types, ctx_map_dev_ids = bind_ctx_param group2ctx
      shared_handle = shared_exec.nil? ? Pointer(Void).null : shared_exec.handle.to_unsafe
      MXNet.check_call LibMXNet.mx_executor_bind_ex(@handle,
        ctx.device_type.value,
        ctx.device_id,
        ctx_map_keys.size,
        ctx_map_keys,
        ctx_map_dev_types,
        ctx_map_dev_ids,
        args_handle.size,
        args_handle,
        args_grad_handle.size != 0 ? args_grad_handle.to_unsafe : Pointer(LibMXNet::NDArrayHandle).null,
        reqs_array,
        aux_args_handle.size,
        aux_args_handle.to_unsafe,
        shared_handle,
        out exec_handle
      )
      executor = Executor.new exec_handle, self.clone
      executor.arg_arrays = args_ndarray
      executor.grad_arrays = args_grad_ndarray
      executor.aux_arrays = aux_states_ndarray
      executor.ctx = Context.new ctx.device_type, ctx.device_id
      executor.grad_req = reqs_array_
      executor.group2ctx = if group2ctx.nil?
                             nil
                           else
                             group2ctx.map { |k, v| {k, Context.new(v.device_type, v.device_id)} }.to_h
                           end
      executor
    end

    def to_json(io)
      MXNet.check_call LibMXNet.mx_symbol_save_to_json(@handle, out js)
      io << String.new(js)
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
        attr["__shape__"] = Shape.to_str(shape)
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
