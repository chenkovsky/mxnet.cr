module MXNet
  module Data
    abstract class Iter
      include Iterator(Batch)

      abstract def next
      abstract def rewind
      abstract def batch_size

      class Creator
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

        def call(params : Hash(String, String)) : Iter
          keys = params.keys.map &.to_unsafe
          vals = params.values.map &.to_unsafe
          handle = DataIterHandle.null
          MXNet.check_call LibMXNet.mx_data_iter_create_iter(@handle, keys, vals, out handle)
          data_name = params.fetch("data_name", "data")
          label_name = params.fetch("label_name", "label")
          MXDataIter.new(handle, data_name, label_name)
        end

        alias IterCreateFunc = Hash(String, String) -> Iter
        @@iter_create_funcs : Hash(Type, Creator) = init_creator.index_by { |x| Type.parse x.name }

        def self.[](t : Type)
          @@iter_creator[t]
        end

        def self.init_creator : Array(Creator)
          MXNet.check_call LibMXNet.mx_list_data_iters(out size, out iter_creator)
          (0...size).map do |idx|
            iter_creator_func = iter_creator[idx]
            MXNet.check_call LibMXNet.mx_data_iter_get_iter_info(iter_creator_func, out name, out desc,
              out num_args, out arg_names, out arg_types, out arg_descs)
            args = (0...num_args).map { |arg_idx|
              Argument.new String.new(arg_names[arg_idx]), String.new(arg_types[arg_idx]), String.new(arg_descs[arg_idx])
            }
            Creator.new iter_creator_func, String.new(name), String.new(desc), args
          end
        end

        # 给每个数组分配名字
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
      end

      class MXIter < Iter
        include Iterator(Batch)
        @handle : LibMXNet::DataIterHandle
        @data_name : String
        @label_name : String
        @current_batch : Batch?
        @provide_data : Hash(String, Shape)?
        @provide_label : Hash(String, Shape)?
        @batch_size : Int32

        getter :batch_size

        def initialize(@handle, @data_name = "data", @label_name = "label")
          tp = if iter_next
                 data = @current_batch.as(Batch).data[0]
                 label = @current_batch.as(Batch).label[0]
                 rewind
                 { {@data_name => data.shape}, {@label_name => label.shape}, data.shape[0] }
               else
                 {nil, nil, 0}
               end
          @provided_data, @provided_label, @batch_size = tp
        end

        def next : Batch
          if @current_batch.nil?
            iter_next
          end
          if !@current_batch.nil?
            batch = @current_batch
            @current_batch = nil
            batch
          else
            stop
          end
        end

        private def iter_next : Bool
          next_ = 0
          MXNet.check_call LibMXNet.mx_data_iter_next(@handle, out next_)
          @current_batch = nil
          if next_ > 0
            @current_batch = Batch.new(get_data, get_label, get_index, get_pad)
          end
        end

        private def get_data : Array(NDArray)
          out_ = NDArrayHandle.null
          MXNet.check_call LibMXNet.mx_data_iter_get_data(@handle, out out_)
          return [NDArray.new(out_, writable: false)]
        end

        private def get_label : Array(NDArray)
          out_ = NDArrayHandle.null
          MXNet.check_call LibMXNet.mx_data_iter_get_label(@handle, out out_)
          return [NDArray.new(out_, writable: false)]
        end

        private def get_index : Array(UInt64)
          out_idx = Pointer(UInt64).null
          out_size = 0_i64
          MXNet.check_call LibMXNet.mx_data_iter_get_index(@handle, out out_index, out out_size)
          (0...out_size).map { |idx| out_idx[idx] }
        end

        private def get_pad : MXUInt
          out_ = MXUInt.new 0
          MXNet.check_call LibMXNet.mx_data_iter_get_pad_num(@handle, out out_)
          return out_
        end

        def rewind
          @current_batch = nil
          MXNet.check_call LibMXNet.mx_data_iter_before_first(@handle)
        end
      end

      class NDArrayIter < Iter
        include Iterator(Batch)
        @data : Array(NDArray)
        @label : Array(NDArray)?
        @batch_size : Int32
        @shuffle : Bool
        @last_batch_handle : LastBatchHandle
        @data_list : Array(NDArray)
        @label_list : Array(NDArray)?
        @init_data : Array({String, NDArray})
        @init_label : Array({String, NDArray})
        @num_data : MXUInt
        @num_source : Int32
        @cursor : Int32
        @init_data : Hash(String, Shape)
        @init_label : Hash(String, Shape)
        getter :batch_size
        enum LastBatchHandle
          Pad
          Discard
          RollOver
        end

        def initialize(@data, @label = nil, @batch_size = 1, @shuffle = false, @last_batch_handle = LastBatchHandle::Pad)
          raise MXError.new("data.size should not be zero") if @data.size == 0
          raise MXError.new("shuffle is not supported currently") if @shuffle
          raise MXError.new("data.size must be equal to label.size") if !@label.nil? && @label.as(Array(NDArray)).size != @data.size
          @data_list, @label_list = case @last_batch_handle
                                    when LastBatchHandle::Discard
                                      data_size = data[0].shape[0]
                                      raise MXError.new "batch_size need to be smaller than data size when not padding." unless @batch_size <= data_size
                                      keep_size = data_size - data_size % data_batch_size
                                      data_list = @data.map { |arr| arr[0, keep_size] }
                                      label_list = if @label.nil?
                                                     [] of NDArray
                                                   else
                                                     @label.as(Array(NDArray)).map { |arr| arr[0, keep_size] }
                                                   end
                                      {data_list, label_list}
                                    else
                                      {@data, @label}
                                    end
          @init_data = IO.init_data(@data_list, false, "data")
          @init_label = IO.init_data(@label_list, true, "label")
          @num_data = @data_list[0].shape[0]
          @num_source = @init_data.size
          @cursor = -@batch_size
          @provide_data = @init_data.map { |name, arr| {name, arr.shape} }.to_h
          @provide_label = @init_label.map { |name, arr| {name.arr.shape} }.to_h
        end

        # get shape via dataBatchSize
        private def get_shape(arr : NDArray)
          s = arr.shape.shape.clone
          s[0] = @batch_size
          Shape.new s
        end

        def hard_rewind
          @cursor = -@batch_size
        end

        def rewind
          if @last_batch_handle == LastBatchHandle::RollOver && @cursor > @num_data
            @cursor = -@batch_size + (@cursor % @num_data) % @batch_size
          else
            @cursor = -@batch_size
          end
        end

        def next : Batch
          if @cursor + @batch_size < @num_data
            @cursor += data_batch_size
            Batch.new get_data, get_label, get_index, get_pad
          else
            stop
          end
        end

        private def pad_data(arr : NDArray) : NDArray
          pad_num = @cursor + @batch_size - @num_data
          shape = arr.shape.shape.clone
          shape[0] = @batch_size
          new_arr = NDArray.zeros(s)
          new_arr[0...(@batch_size - @pad_num)] = arr[@cursor, @num_data]
          new_arr[@batch_size - @pad_num, @batch_size] = arr[0, @pad_num]
          new_arr
        end

        private def get_data
          raise MXError.new "cursor >= num_data, Iter needs rewind" unless @cursor < @num_data
          if @cursor + @batch_size <= @num_data
            @data_list.map { |arr| arr[@cursor, @batch_size] }
          else
            @data_list.map { |arr| pad_data arr }
          end
        end

        private def get_pad
          if @last_batch_handle == LastBatchHandle::Pad && @cursor + @batch_size > @num_data
            @cursor + @batch_size - @num_data
          else
            0
          end
        end
      end

      class PrefecthingIter < Iter
        @iters : Array(Iter)
        @data_names : Array(Hash(String, String))?
        @label_names : Array(Hash(String, String))?
        @provide_data : Hash(String, Shape)
        @provide_label : Hash(String, Shape)
        @batch_size : Int32
        @current_batch : Batch?
        @next_batch : Array(Batch?)

        getter :batch_size

        def initialize(@iters, @data_names = nil, @label_names = nil)
          raise MXError.new "iters length must be greater than 0" unless iters.size > 0
          @provide_data = {} of String => Shape
          @provide_label = {} of String => Shape
          if @data_names.nil?
            @iters.each do |iter|
              @provide_data.merge! iter.provide_data
            end
          else
            @iters.each_with_index do |iter, idx|
              iter.provide_data.each do |k, v|
                @provide_data[@data_names[idx][k]] = v
              end
            end
          end
          if @label_names.nil?
            @iters.each do |iter|
              @provide_label.merge! iter.provide_label
            end
          else
            @iters.each_with_index do |iter, idx|
              iter.provided_label.each do |k, v|
                @label_names[@data_names[idx][k]] = v
              end
            end
          end
          @batch_size = @provide_data.first_value[0]
          @started = true
          @current_batch = nil
          @next_batch = Array(Batch?).new(iters.size) { |i| Batch.new nil, nil, nil, 0 }
          prefetch_func = ->(i) {
            while @started
              inst = iters[i].next
              if inst == Iterator::Stop::INSTANCE
                next_batch[i] = nil
              else
                next_batch[i] = inst
              end
            end
          }
          prefecth_threads = (0...iters.size).each do |i|
            spawn do
              prefetch(i)
            end
          end
        end

        def next
          if @current_batch.nil?
            iter_next
          end
          if @current_batch.nil?
            stop
          else
            @current_batch.as(Batch)
          end
        end

        private def iter_next
          if @next_batch[0].nil?
            @next_batch.each do |batch|
              raise MXError.new "Number of entry mismatches between iterators" unless batch.nil?
            end
            return false
          end
          @next_batch.each do |batch|
            raise MXError.new "Number of entry mismatches between iterators" if batch.nil?
          end
          datas = [] of NDArray
          @next_batch.each { |b| b.data.each { |d| datas << d } }
          labels = [] of NDArray
          @next_batch.each { |b| b.label.each { |d| labels << d } }
          @current_batch = Batch.new(datas, labels, @next_batch[0].index, @next_batch[0].pad)
          true
        end

        def rewind
          @iters.each do |iter|
            iter.rewind
          end
        end

        def get_data
          @current_batch.as(Batch).data
        end

        def get_label
          @current_batch.as(Batch).label
        end

        def get_index
          @current_batch.as(Batch).index
        end

        def get_pad
          @current_batch.as(Batch).pad
        end
      end
    end
  end
end
