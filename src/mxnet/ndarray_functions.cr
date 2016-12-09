module MXNet
  class NDArray::Function
    enum FunctionType
      Binary
      Unary
      Generic
    end

    class Argument
      @name : String
      @type : String
      @description : String

      def initialize(@name, @type, @description)
      end

      def to_s(io)
        io << "@param #{@name} : #{@type} # #{@description}"
      end
    end

    def accept_empty_mutate?
      @accept_empty_mutate
    end

    getter :name, :func_type, :handle
    @handle : LibMXNet::FunctionHandle
    @name : String
    @description : String
    @args : Array(Argument)
    @return_type : String
    @use_vars_range : Range(Int32, Int32)
    @scalar_range : Range(Int32, Int32)
    @accept_empty_mutate : Bool
    @num_mutate_vars : Int32
    @func_type : FunctionType
    private NDARRAY_ARG_BEFORE_SCALAR = 1
    private ACCEPT_EMPTY_MUTATE_TARGET = 1 << 2

    def initialize(@handle, @name, @description,
                   @args, @return_type, num_use_vars, num_scalars, @num_mutate_vars, type_mask)
      @accept_empty_mutate = (type_mask & ACCEPT_EMPTY_MUTATE_TARGET) != 0

      ndarray_arg_before_scalar = (type_mask & NDARRAY_ARG_BEFORE_SCALAR) != 0
      @use_vars_range = if ndarray_arg_before_scalar
                          (0...num_use_vars)
                        else
                          (num_scalars...(num_scalars + num_use_vars))
                        end
      @scalar_range = if ndarray_arg_before_scalar
                        (num_use_vars...(num_use_vars + num_scalars))
                      else
                        (0...num_scalars)
                      end
      @func_type = if num_mutate_vars == 1 && num_use_vars == 2 && num_scalars == 0
                     Binary
                   elsif num_mutate_vars == 1 && num_use_vars == 1 && num_scalars == 0
                     Unary
                   else
                     Generic
                   end
    end

    def to_s(io)
      io << "@func #{@name}\n"
      io << "@return #{@return_type}\n"
      io << "##{@description}\n"
      @args.each do |arg|
        arg.to_s io
        io << "\n"
      end
    end

    private def self.add_dependency(froms : Array(NDArray), tos : Array(NDArray))
      froms.each do |f|
        tos.each do |t|
          t.dependencies << from
          t.dependencies.concat from
        end
      end
    end

    def self.invoke_binary(func : Function, lhs : NDArray, rhs : NDArray, out out_ : NDArray? = nil) : NDArray
      raise MXError.new "out must be writable" unless output.nil? || output.writable?
      raise MXError.new "call #{func.name} as binary function" if func.func_type != FunctionType::Binary
      output = if out_.nil?
                 raise MXError.new "argument out is required to call #{func.name}" unless func.accept_empty_mutate?
                 arr = NDArray.empty
                 add_dependency [lhs, rhs], [arr]
                 arr
               else
                 out_
               end

      check_call(LibMXNet.mx_func_invoke(func.handle, [lhs.handle, rhs.handle], [] of MXFloat, [output.handle]))
      return output
    end

    def self.invoke_unary(func : Function, src : NDArray, out out_ : NDArray? = nil) : NDArray
      raise MXError.new "out must be writable" unless output.nil? || output.writable?
      raise MXError.new "call #{func.name} as unary function" if func.func_type != FunctionType::Unary
      output = if out_.nil?
                 raise MXError.new "argument out is required to call #{func.name}" unless func.accept_empty_mutate?
                 arr = NDArray.empty
                 add_dependency [src], [arr]
                 arr
               else
                 out_
               end
      check_call(LibMXNet.mx_func_invoke(func.handle, [src.handle], [] of MXFloat, [output.handle]))
    end

    def invoke_generic(func : Function, *args, **kwargs) : NDArray
      raise MXError.new "call #{func.name} as generic function" if func.func_type != FunctionType::Generic
      if !kwargs.nil?
        kwargs_h = kwargs.to_h
        if kwargs.has_key? :out
          out_ = kwargs[:out]
          mutate_vars = if out_.is_a? NDArray
                          out_.as(NDArray)
                        else
                          out_.as(Array(NDArray))
                        end
          kwargs_h.reject! :out
        end
      else
        mutate_vars = nil
        kwargs_h = nil
      end
      raise MXError.new "expect #{func.num_mutate_vars} in #{func.name}" unless mutate_vars.nil? || mutate_vars.length == func.num_mutate_vars
      use_vars = func.use_vars_range.map { |x| args[x].as(NDArray) }
      scalar_vars = func.scalar_range.map { |x| args[x].as(MXFloat) }
      outputs = if mutate_vars.nil?
                  raise MXError.new "argument out is required to call #{func.name}" unless func.accept_empty_mutate?
                  arr = Array(NDArray).new(func.num_mutate_vars) { |idx| NDArray.empty }
                  add_dependency use_vars, arr
                  arr
                else
                  mutate_vars
                end
      num_kwargs, kwarg_keys, kwarg_vals = if kwargs_h.nil?
                                             {0, nil, nil}
                                           else
                                             {kwargs_h.size,
                                               kwargs_h.keys.map { |x| x.to_unsafe },
                                               kwargs_h.values.map { |x| x.to_s.to_unsafe }}
                                           end
      check_call(LibMXNet.mx_func_invoke_ex(
        use_vars.map { |x| x.handle },
        scalar_vars,
        outpus.map { |x| x.handle },
        num_kwargs, kwarg_keys, kwarg_vals
      ))
      outputs
    end

    private def self.init_functions
      function_num = MXUInt.new 0
      function_handles = Pointer(FunctionHandle).null
      check_call(LibMXNet.mx_list_functions(out function_num, out function_handles))
      functions = (0...function_num).map do |idx|
        f = function_handles[idx]
        name = Pointer(UInt8).null
        description = Pointer(UInt8).null
        num_args = MXUInt.new 0
        arg_names = Pointer(Pointer(UInt8)).null
        arg_type_infos = Pointer(Pointer(UInt8)).null
        arg_descriptions = Pointer(Pointer(UInt8)).null
        return_type = Pointer(UInt8).null
        check_call(LibMXNet.mx_func_get_info(f, out name, out description,
          out num_args, out arg_names,
          out arg_type_infos, out arg_descriptions,
          out return_type
        ))
        num_use_vars = MXUInt.new 0
        num_scalars = MXUInt.new 0
        num_mutate_vars = MXUInt.new 0
        type_mask = Int32.new 0
        check_call(LibMXNet.mx_func_describe(f, out num_use_vars, out num_scalars, out num_mutate_vars, out type_mask))
        args = (0...num_args).map do |arg_idx|
          Argument.new String.new(arg_names[arg_idx]), String.new(arg_type_infos[arg_idx]), String.new(arg_descriptions[arg_idx])
        end
        Function.new f, String.new(name), String.new(description), args, String.new(return_type), num_use_vars, num_scalars, num_mutate_vars, type_mask
      end
      functions.index_by &.name
    end

    Functions = init_functions

    def self.[](name : String)
      Functions[name]
    end

    macro def_functions(*names)
        {% for name, index in names %}
        F_{{name.id}} = Functions["_{{name.id}}"]
        {% end %}
      end

    def_functions :onehot_encode, :clip, :sqrt, :rsqrt, :dot
    def_functions :norm, :abs, :sign, :round, :ceil, :floor
    def_functions :square, :exp, :log, :cos, :sin, :max, :min
    def_functions :sum, :argmax_channel, :choose_element_0index
    def_functions :sample_uniform, :sample_normal, :set_value
    def_functions :plus, :plus_scalar, :minus, :_minus_scalar, :mul
    def_functions :mul_scalar, :div, :div_scalar
    def_functions :rminus_scalar, :rdiv_scalar
  end

  class NDArray
    macro def_unary(*names)
        {% for name, index in names %}
        def {{name.id}} : NDArray
            Function.invoke_unary(Function::F_{{name.id}}, self)
        end
        {% end %}
    end

    macro def_binary(*names)
        {% for name, index in names %}
        def {{name.id}}(rhs) : NDArray
            Function.invoke_binary(Function::F_{{name.id}}, self, rhs)
        end
        {% end %}
    end

    def_unary :sqrt, :rsqrt, :norm, :abs, :sign, :round, :ceil, :floor, :square
    def_unary :exp, :log, :cos, :sin, :max, :min, :sum, :argmax_channel
    def_binary :plus, :minus, :mul, :div
    def_binary :dot, :choose_element_0index

    def onehotEncode(out out_ : NDArray) : NDArray
      Function.invoke_binary(Function::F_onehot_encode, self, out_, out_)
    end

    def clip(min : Float64, max : Float64) : NDArray
      Function.invoke_generic(Function::F_clip, self, min, max)[0]
    end

    def random_uniform(low : Float64, high : Float64) : NDArray
      Function.invoke_generic(Function::F_sample_uniform, low: low, high: high, shape: shape, out: self)[0]
    end

    def random_gaussian(loc : Float64, scale : Float64) : NDArray
      Function.invoke_generic(Function::F_sample_normal, loc: loc, scale: scale, shape: shape, out: self)[0]
    end

    def set(value : Float64) : NDArray
      Function.invoke_generic(Function::F_set_value, value, out: self)
      self
    end

    def set(rhs : NDArray) : NDArray
      rhs.copy_to(self)
    end

    def set(rhs : Array(Float64)) : NDArray
      sync_copy_from(rhs)
      self
    end

    def +(rhs : NDArray) : NDArray
      plus(rhs)
    end

    def plus!(rhs : NDArray) : NDArray
      Function.invoke_binary(Function::F_plus, self, rhs, out: self)
    end

    def -(rhs : NDArray) : NDArray
      minus(rhs)
    end

    def -(rhs : Float) : NDArray
      Function.invoke_generic(Function::F_minus_scalar, self, rhs)[0]
    end

    def minus!(rhs : NDArray) : NDArray
      Function.invoke_binary(Function::F_minus, self, rhs, out: self)
    end

    def minus!(rhs : Float) : NDArray
      Function.invoke_generic(Function::F_minus_scalar, self, rhs, out: self)[0]
    end

    def *(rhs : NDArray) : NDArray
      mul(rhs)
    end

    def *(rhs : Float64) : NDArray
      Function.invoke_generic(Function::F_mul_scalar, self, rhs)[0]
    end

    def mul!(rhs : NDArray) : NDArray
      Function.invoke_binary(Function::F_mul, self, rhs, out: self)
    end

    def mul!(rhs : Float64) : NDArray
      Function.invoke_generic(Function::F_mul_scalar, self, rhs, out: self)[0]
    end

    def - : NDArray
      mul -1.0
    end

    def /(rhs : NDArray) : NDArray
      div(rhs)
    end

    def /(rhs : Float64) : NDArray
      Function.invoke_generic(Function::F_div_scalar, self, rhs)[0]
    end

    def div!(rhs : NDArray) : NDArray
      Function.invoke_binary(Function::F_div, self, rhs, out: self)
    end

    def div!(rhs : Float64) : NDArray
      Function.invoke_generic(Function::F_div_scalar, self, rhs, out: self)[0]
    end

    private def _copy_to(other : NDArray) : NDArray
      Function.invoke_unary(Function::F_copyto, self, out: other)
    end
  end
end

struct Float
  def +(other : MXNet::NDArray)
    other + self
  end

  def -(other : NDArray)
    MXNet::Function.invoke_generic(MXNet::Function::F_rminus_scalar, other, self)[0]
  end

  def *(other : NDArray)
    other * self
  end

  def /(other : NDArray)
    MXNet::Function.invoke_generic(MXNet::Function::F_rdiv_scalar, other, self)[0]
  end
end