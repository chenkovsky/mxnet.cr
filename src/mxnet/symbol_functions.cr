module MXNet
  class Symbol::Function
    enum BindReq
      NULL  = 0
      WRITE = 1
      ADD   = 3
    end
    getter :name

    Functions = Hash(String, Symbol::Function).new
    init_functions.each do |f|
      Functions[f.name] = f
      STDERR.puts f.to_s
      STDERR.puts "================"
    end

    macro def_functions(*names)
        {% for name, index in names %}
        F_{{name.id}} = Functions["{{name.id}}"]
        {% end %}
    end

    def_functions :Activation

    class Argument
      def initialize(@name : String, @type : String, @desc : String)
      end

      def to_s(io)
        io << "@param #{@name} : #{@type} ##{@desc}"
      end
    end

    def to_unsafe
      @handle
    end

    getter :name
    getter :key_var_num_args
    @key_var_num_args : String?

    def initialize(@handle : LibMXNet::AtomicSymbolCreator, @name : String, @desc : String, @ret_type : String, key_var_num_args : String?, @arguments : Array(Argument))
      @key_var_num_args = key_var_num_args.nil? || key_var_num_args.size == 0 ? nil : key_var_num_args
    end

    def to_s(io)
      io << "@func #{@name}\n"
      io << "@return #{@ret_type}\n"
      io << "@key_var_num_args #{@key_var_num_args}\n"
      io << "##{@desc}\n"
      @arguments.each do |arg|
        arg.to_s io
        io << "\n"
      end
    end

    def num_args
      @arguments.size
    end

    private def self.init_functions
      MXNet.check_call(LibMXNet.mx_symbol_list_atomic_symbol_creators(out list_size, out symbol_list))
      (0...list_size).map do |idx|
        handle = symbol_list[idx]
        MXNet.check_call(LibMXNet.mx_symbol_get_atomic_symbol_info(handle,
          out name,
          out desc,
          out num_args,
          out arg_names,
          out arg_types,
          out arg_descs,
          out key_var_num_args,
          out ret_type))
        name_s = String.new name
        desc_s = String.new desc
        args = (0...num_args).map do |idx|
          arg_name = String.new arg_names[idx]
          arg_type = String.new arg_types[idx]
          arg_desc = String.new arg_descs[idx]
          arg = Symbol::Function::Argument.new arg_name, arg_type, arg_desc
        end
        Symbol::Function.new handle, String.new(name), String.new(desc), String.new(ret_type), String.new(key_var_num_args), args
      end
    end
  end

  class Symbol
    enum ActType
      Relu
      Sigmoid
      Tanh
      Softrelu
    end

    def activation(act_type : ActType = ActType::Relu, name : String? = nil)
      create(Function::F_Activation, name: name, act_type: act_type.to_s.downcase)
    end
  end
end
