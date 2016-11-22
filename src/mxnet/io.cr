module MXNet
  module IO
    class DataIterCreator
      class Argument
        @name : String
        @type : String
        @desc : String

        def initialize(@name, @type, @desc)
        end
      end

      @name : String
      @desc : String
      @args : Array(Argument)
      @handle : LibMXNet::DataIterCreator
      getter :name

      def initialize(@handle, @name, @desc, @args)
      end

      def call(params : Hash(String, String))
        keys = params.keys.map &.to_unsafe
        vals = params.values.map &.to_unsafe
        handle = DataIterHandle.null
        check_call LibMXNet.mx_data_iter_create_iter(@handle, keys, vals, out handle)
        data_name = params.fetch("data_name", "data")
        label_name = params.fetch("label_name", "label")
        MXDataIter.new(handle, data_name, label_name)
      end
    end

    alias IterCreateFunc = Hash(String, String) -> DataIter
    alias PackCreateFunc = Hash(String, String) -> DataPack
    @@iter_create_funcs : Hash(String, DataIterCreator) = init_io_module.index_by &.name

    def self.init_io_module : Array(DataIterCreator)
      iter_creator = Pointer(DataIterCreator)
      size = MXUInt.new 0
      check_call LibMXNet.mx_list_data_iters(out size, out iter_creator)
      (0...size).map do |idx|
        iter_creator_func = iter_creator[idx]
        name = Pointer(UInt8).null
        desc = Pointer(UInt8).null
        arg_names = Pointer(Pointer(UInt8)).null
        arg_types = Pointer(Pointer(UInt8)).null
        arg_descs = Pointer(Pointer(UInt8)).null
        num_args = MXUInt.new 0
        check_call LibMXNet.mx_data_iter_get_iter_info(iter_creator, out name, out desc,
          out num_args, out arg_names, out arg_types, out arg_descs)
        args = (0...num_args).map { |arg_idx|
          Argument.new String.new(arg_names[arg_idx]), String.new(arg_types[arg_idx]), String.new(arg_descs[arg_idx])
        }
        DataIterCreator.new iter_creator_func, name, desc, args
      end
    end

    def self.init_data(data : Array(NDArray)?, allow_empty : Bool, default_name : String) : Array({String, NDArray})
      raise MXError.new "!data.nil? || allow_empty is not satisfied" unless !data.nil? || allow_empty
      if data.nil?
        [] of {String, NDArray}
      elsif data.size == 1
        [{default_name, data[0]}]
      else
        data.map_with_index do |e, i|
          {"#{default_name}_#{i}", e}
        end
      end
    end

    class DataBatch
      @data : Array(NDArray)
      @label : Array(NDArray)
      @index : Array(Int64)
      @pad : Int32
      @bucket_key : BucketKey?
      @provided_data : Hash(String, Shape)?
      @provided_label : Hash(String, Shape)?

      def initialize(@data, @label, @index, @pad, @bucket_key = nil, @provided_data = nil, @provided_label = nil)
      end
    end

    abstract class DataIter
      include Enumerable(DataBatch)

      abstract def each
      abstract def pad
      abstract def data
      abstract def batch_size
      abstract def label
      abstract def index
      abstract def provided_label
      abstract def provided_data
      abstract def default_bucket_key
    end

    abstract class DataPack
      include Enumerable(DataBatch)

      abstract def iter : DataIter
    end
  end
end
