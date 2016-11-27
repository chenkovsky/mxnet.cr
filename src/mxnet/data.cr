module MXNet
  module Data
    enum Type
      MNIST
      ImageRecord
      CSV
    end

    class Batch
      abstract class BucketKey
      end

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

    abstract class BasePack
      include Iterable(Batch)

      abstract def each
    end

    class Pack < BasePack
      def initialize(@iter_type : Type, @params : Hash(String, String))
      end

      def each
        IterCreator[t].call(@params)
      end

      macro def_pack(name, typ_)
        def {{name.id}}(params : Hash(String, String))
          Pack.new {{typ_}}, params
        end
      end

      def_pack :mnist, Type::MNIST
      def_pack :image_record, Type::ImageRecord
      def_pack :csv, Type::CSV
    end
  end
end
