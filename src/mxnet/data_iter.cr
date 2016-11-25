module MXNet
  class MXDataIter
    include Iterator(DataBatch)
    @handle : DataIterHandle
    @data_name : String
    @label_name : String
    @current_batch : DataBatch?
    @provide_data : Hash(String, Shape)
    @provide_label : Hash(String, Shape)
    @batch_size : Int32

    def initialize(@handle, @data_name = "data", @label_name = "label")
      tp = if iter_next
             data = @current_batch.as(DataBatch).data[0]
             label = @current_batch.as(DataBatch).label[0]
             rewind
             { {@data_name => data.shape}, {@label_name => label.shape}, data.shape[0] }
           else
             {nil, nil, 0}
           end
      @provided_data, @provided_label, @batch_size = tp
    end

    def next : DataBatch
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
      check_call LibMXNet.mx_data_iter_next(@handle, out next_)
      @current_batch = nil
      if next_ > 0
        @current_batch = DataBatch.new(get_data, get_label, get_index, get_pad)
      end
    end

    private def get_data : Array(NDArray)
      out_ = NDArrayHandle.null
      check_call LibMXNet.mx_data_iter_get_data(@handle, out out_)
      return [NDArray.new(out_, writable: false)]
    end

    private def get_label : Array(NDArray)
      out_ = NDArrayHandle.null
      check_call LibMXNet.mx_data_iter_get_label(@handle, out out_)
      return [NDArray.new(out_, writable: false)]
    end

    private def get_index : Array(UInt64)
      out_idx = Pointer(UInt64).null
      out_size = 0_i64
      check_call LibMXNet.mx_data_iter_get_index(@handle, out out_index, out out_size)
      (0...out_size).map { |idx| out_idx[idx] }
    end

    private def get_pad : MXUInt
      out_ = MXUInt.new 0
      check_call LibMXNet.mx_data_iter_get_pad_num(@handle, out out_)
      return out_
    end

    def rewind
      @current_batch = nil
      check_call LibMXNet.mx_data_iter_before_first(@handle)
    end
  end

  class NDArrayIter
    include Iterator(DataBatch)
    @data : Array(NDArray)
    @label : Array(NDArray)?
    @data_batch_size : Int32
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
    enum LastBatchHandle
      Pad
      Discard
      RollOver
    end

    def initialize(@data, @label = nil, @data_batch_size = 1, @shuffle = false, @last_batch_handle = LastBatchHandle::Pad)
      raise MXError.new("data.size should not be zero") if @data.size == 0
      raise MXError.new("shuffle is not supported currently") if @shuffle
      raise MXError.new("data.size must be equal to label.size") if !@label.nil? && @label.as(Array(NDArray)).size != @data.size
      @data_list, @label_list = case @last_batch_handle
                                when LastBatchHandle::Discard
                                  data_size = data[0].shape[0]
                                  raise MXError.new "batch_size need to be smaller than data size when not padding." unless @data_batch_size <= data_size
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
      @cursor = -@data_batch_size
      @provide_data = @init_data.map { |name, arr| {name, arr.shape} }.to_h
      @provide_label = @init_label.map { |name, arr| {name.arr.shape} }.to_h
    end

    # get shape via dataBatchSize
    private def get_shape(arr : NDArray)
      s = arr.shape.shape.clone
      s[0] = @data_batch_size
      Shape.new s
    end

    def hard_rewind
      @cursor = -@data_batch_size
    end

    def rewind
      if @last_batch_handle == LastBatchHandle::RollOver && @cursor > @num_data
        @cursor = -@data_batch_size + (@cursor % @num_data) % @data_batch_size
      else
        @cursor = -@data_batch_size
      end
    end

    def next : DataBatch
      if @cursor + @data_batch_size < @num_data
        @cursor += data_batch_size
        DataBatch.new get_data, get_label, get_index, get_pad
      else
        stop
      end
    end

    private def pad_data(arr : NDArray) : NDArray
      pad_num = @cursor + @data_batch_size - @num_data
      shape = arr.shape.shape.clone
      shape[0] = @data_batch_size
      new_arr = NDArray.zeros(s)
      new_arr[0...(@data_batch_size - @pad_num)] = arr[@cursor, @num_data]
      new_arr[@data_batch_size - @pad_num, @data_batch_size] = arr[0, @pad_num]
      new_arr
    end

    private def get_data
      raise MXError.new "cursor >= num_data, DataIter needs rewind" unless @cursor < @num_data
      if @cursor + @data_batch_size <= @num_data
        @data_list.map { |arr| arr[@cursor, @data_batch_size] }
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

  class PrefecthingIter
    @iters : Array(DataIter)
    @data_names : Array(Hash(String, String))?
    @label_names : Array(Hash(String, String))?
    @provide_data : Hash(String, Shape)
    @provide_label : Hash(String, Shape)
    @batch_size : Int32
    @current_batch : DataBatch?
    @next_batch : Array(DataBatch?)

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
      @next_batch = Array(DataBatch?).new(iters.size) { |i| DataBatch.new nil, nil, nil, 0 }
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
        @current_batch.as(DataBatch)
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
      @current_batch = DataBatch.new(datas, labels, @next_batch[0].index, @next_batch[0].pad)
      true
    end

    def rewind
      @iters.each do |iter|
        iter.rewind
      end
    end

    def get_data
      @current_batch.as(DataBatch).data
    end

    def get_label
      @current_batch.as(DataBatch).label
    end

    def get_index
      @current_batch.as(DataBatch).index
    end

    def get_pad
      @current_batch.as(DataBatch).pad
    end
  end
end
