module MXNet
  class Symbol::Function
    enum BindReq
      NULL  = 0
      WRITE = 1
      ADD   = 3
    end
    getter :name

    @@functions = Hash(String, Symbol::Function).new
    init_functions.each do |f|
      @@functions[f.name] = f
    end

    def self.[](name)
      @@functions[name]
    end

    class Argument
      def initialize(@name : String, @type : String, @desc : String)
      end
    end

    def initialize(@handle : LibMXNet::AtomicSymbolCreator, @name : String, @desc : String, @ret_type : String, @key_var_num_args : String, @arguments : Array(Argument))
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
end
