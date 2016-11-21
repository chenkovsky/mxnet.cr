module MXNet
  # Model class of MXNet for training and predicting feedforward nets.
  # This class is designed for a single-data single output supervised network.
  # @param symbol The symbol configuration of computation network.
  # @param symGen Symbol generator for bucketing.
  # @param ctx The device context of training and prediction.
  #            To use multi GPU training, pass in a list of gpu contexts.
  # @param numEpoch Training parameter, number of training epochs(epochs).
  # @param epochSize Number of batches in a epoch. In default, it is set to
  #                  ceil(num_train_examples / batch_size)
  # @param optimizer Training parameter, name or optimizer object for training.
  # @param initializer Training parameter, the initialization scheme used.
  # @param batchSize The batch size of training data.
  # @param argParams Model parameter, dict of name to NDArray of net's weights.
  # @param auxParams Model parameter, dict of name to NDArray of net's auxiliary states.
  # @param allowExtraParams Whether allow extra parameters that are not needed by symbol
  #                         to be passed by aux_params and arg_params.
  #                         If this is True, no error will be thrown when aux_params and arg_params
  #                         contain extra parameters than needed.
  # @param beginEpoch The beginning training epoch.
  class FeedForward
    @symbol : Symbol
    @sym_gen : SymbolGenerator?
    @ctx : Array(Context)
    @num_epoch : Int32
    @epoch_size : Int32
    @optimizer : Optimizer
    @initializer : Initializer
    @batch_size : Int32
    @arg_params : Hash(String, NDArray)?
    @aux_params : Hash(String, NDArray)?
    @allow_extra_params : Bool
    @begin_epoch : Int32
    @argument_checked : Bool
    @pred_exec : Executor?
    @monitor : Monitor?
    getter :arg_params, :aux_params

    def initialize(@symbol, @sym_gen = nil, @ctx = [Context.cpu], @num_epoch = -1, @epoch_size = -1,
                   @optimizer = SGD.new, @initializer = Uniform.new(0.01_f32), @batch_size = 128,
                   @arg_params = nil, @aux_params = nil, @allow_extra_params = false, @begin_epoch = 0)
      @argument_checked = false
      if @sym_gen.nil?
        check_arguments
      end
    end

    # verify the argument of the default symbol and user provided parameters
    def check_arguments
      if !@argument_checked
        ExecutorManager.check_arguments @symbol
        if @allow_extra_params
          if !@arg_params.nil?
            arg_names = @symbol.arguments.to_set
            @arg_params.select! { |k, v| arg_names.includes? k }
          end
          if !@aux_params.nil?
            aux_names = @symbol.auxiliary_states.to_set
            @aux_params.select! { |k, v| aux_names.includes? k }
          end
        end
        @argument_checked = true
      end
    end

    setter :monitor
    private def init_params(input_shapes : Hash(String, Shape), overwrite : Bool = false)
      arg_shapes, _, aux_shapes = @symbol.infer_shape input_shapes
      arg_names = @symbol.arguments
      param_names = arg_names.select { |x| !input_shapes.has_key? x }
      aux_names = @symbol.auxiliary_states
      param_name_shapes = arg_names.zip(arg_shapes).select { |name, _| param_names.includes? name }
      arg_params = param_name_shapes.map { |name, shape| {name, NDArray.zeros(shape)} }.to_h
      aux_params = aux_names.zip(aux_shapes).map { |name, shape| {name, NDArray.zeros(shape)} }.to_h
      arg_params.each do |k, v|
        if !@arg_params.nil? && @arg_params.has_key?(k) && !overwrite
          arg_params[k].set(@arg_params[k])
        else
          initializer(k, v)
        end
      end
      aux_params.each do |k, v|
        if !@aux_params.nil? && @aux_params.has_key?(k) && !overwrite
          aux_params[k].set(@aux_params[k])
        else
          initializer(k, v)
        end
      end
      @arg_params = arg_params
      @aux_params = aux_params
      return arg_names, param_names, aux_names
    end
    private def init_predictor(input_shapes : Hash(String, Shape))
      if !@pred_exec.nil?
        arg_shapes, _, _ = @symbol.infer_shape input_shapes
        raise MXError.new "Incomplete input shapes" if arg_shapes.nil?
        pred_shapes = @pred_exec.arg_arrays.map { |x| x.shape }
        if arg_shapes.same_elements?(pred_shapes)
          return
        end
      end
      pred_exec = @symbol.bind(ctx[0], grad_req: BindReq::NULL, shapes: input_shapes)
      pred_exec.copy_params_from(@arg_params, @aux_params)
      ExecutorManager.check_arguments @symbol
      @pred_exec = pred_exec
    end
    private def init_iter(x : NDArray, y : NDArray, is_train : Bool) : DataIter
      raise MXError.new "y must be specified" unless !y.nil? || !is_train
      label = if y.nil?
                NDArray.zeros x.shape[0]
              else
                y
              end
      raise MXError.new "Label must be 1D" unless label.shape.size == 1
      raise MXError.new "The numbers of data points and labels not equal" unless x.shape[0] == label.shape[0]
      if is_train
        NDArrayIter.new [x], [label], batch_size, shuffle: is_train, last_batch_handle: LastBatchHandle::ROLL_OVER
      else
        NDArrayIter.new [x], [label], batch_size, shuffle: false
      end
    end
    private def init_eval_iter(eval_data : {NDArray, NDArray}?)
      if eval_data.nil?
        nil
      else
        init_iter(eval_data[0], eval_data[1], is_train: true)
      end
    end

    # Run the prediction, always only use one device.
    # @param data eval data
    # @param numBatch the number of batch to run. Go though all batches if set -1
    # @return The predicted value of the output.
    #         Note the network may have multiple outputs, thus it return an array of [[NDArray]]
    def predict(data : DataIter, num_batch : Int32 = -1) : Array(NDArray)
      data.reset
      data_shapes = data.provide_data
      data_names = data_shapes.map { |x| x[0] }
      init_predictor data_shapes
      batch_size = data.batch_size
      data_arrays = data_names.map { |x| @pred_exec.arg_dict[x] }
      outputs = (0...@pred_exex.outputs.size).map { |x| Array(NDArray).new }
      i = 0
      data.each do |batch|
        break if i == num_batch
        i += 1
        ExecutorManager.load_data(batch, data_arrays)
        @pred_exec.forward(is_train: false)
        padded = batch.pad
        real_size = batch_size - padded
        outputs.zip(pred_exec.outputs) do |lst, nd|
          lst << nd[0, real_size].copy
        end
      end
      results = outputs.map { |x| NDArray.concatenate(x) }
      result
    end

    #  Fit the model.
    # @param trainData Training data
    # @param evalData Evaluation data
    # @param evalMetric The evaluation metric, cannot be null
    # @param epochEndCallback A callback that is invoked at end of each epoch.
    #                         This can be used to checkpoint model each epoch.
    # @param batchEndCallback A callback that is invoked at end of each batch
    #                         For print purpose
    # @param kvStoreType A string kvstore type:
    #                    'local' : multi-devices on a single machine, will automatically
    #                    choose one from 'local_update_cpu', 'local_allreduce_cpu', and
    #                    'local_allreduce_device'
    #                    'dist_sync' : multi-machines with BSP
    #                    'dist_async' : multi-machines with partical asynchronous
    #                    In default uses 'local', often no need to change for single machine.
    # @param logger When not specified, default logger will be used.
    # @param workLoadList The list of work load for different devices, in the same order as ctx
    def fit(train_data : DataIter, eval_data : DataIter, eval_metric : EvalMetric = Accuracy.new, kv_store_type : KVStoreType = KVStoreType::LOCAL, epoch_end_callback : EpochEndCallback? = nil,
            batch_end_callback : BatchEndCallback? = nil,
            logger : Logger = @@logger, work_load_list : Array(Float32)? = nil)
      init_symbol_params train_data
      kv_store, update_on_kv_store = Model.create_kv_store kv
      fit train_data, eval_data, kv_store, update_on_kv_store, eval_metric, epoch_end_callback, batch_end_callback, logger, work_load_list
    end

    private def init_symbol_params(train_data : DataIter)
      if !@sym_gen.nil?
        @symbol = @sym_gen.generate(train_data.default_bucket_key)
        check_arguments
      end
      init_params(train_data.provide_data + train_data.provide_label)
    end
    private def fit(train_data : DataIter, eval_data : DataIter,
                    kv_store : KVStore, update_on_kv_store : Bool,
                    eval_metric : EvalMetric = Accuracy.new,
                    epoch_end_callback : EpochEndCallback? = nil,
                    batch_end_callback : BatchEndCallback? = nil,
                    logger : Logger = @@logger, work_load_list : Array(Float32)? = nil)
      arg_names, param_names, aux_names = init_symbol_params train_data
      batch_size_multiplier = if kv_store.type == kvStoreType
                                ::DIST_SYNC
                                kv_store.num_workers
                              else
                                1
                              end
      batch_size = train_data.batch_size * batch_size_multiplier
      @optimizer.arg_names = arg_names
      @optimizer.rescale_grad = 1_f32/batch_size
      @optimizer.symbol = @symbol
      param_idx2name = if update_on_kv_store
                         param_names.enum_with_index.map { |e_i| e_i }.to_h
                       else
                         arr = [] of {Int32, String}
                         param_names.each_with_index do |e, i|
                           (0...ctx.size).each do |k|
                             arr << {i * ctx.size + k, name}
                           end
                         end
                         arr.to_h
                       end
      @optimizer.idx2name = param_idx2name
      logger.debug("Start training on multi-device")
      Model.train_multi_device @symbol, @ctx, arg_names, param_names, aux_names,
        @arg_params, @aux_params, @begin_epoch, @num_epoch,
        @epoch_size, @optimizer,
        kv_store, update_on_kv_store,
        train_data, eval_data,
        eval_metric,
        epoch_end_callback,
        batch_end_callback,
        work_load_list,
        @monitor,
        @sym_gen
    end

    def save(prefix : String, epoch : UInt32 = @num_epoch)
      Model.save_check_point(prefix, epoch, @symbol, arg_params, aux_params)
    end

    def serialize
      Model.serialize @symbol, arg_params, aux_params
    end

    protected def self.is_data_arg?(name : String)
      name.ends_with?("data") || name.ends_with?("label")
    end

    def self.load(prefix : String, epoch : Int32,
                  ctx : Array(Context) = [Context.cpu],
                  num_epoch : Int32 = -1,
                  epoch_size : Int32 = -1,
                  optimizer : Optimizer = SGD.new,
                  initializer : Initializer = Uniform.new(0.01_f32),
                  batch_size : Int32 = 128,
                  allow_extra_params : Bool = false) : FeedForward
      symbol, arg_params, aux_params = Model.load_check_point prefix, epoch
      FeedForward.new symbol, ctx,
        arg_params, aux_params,
        epoch, num_epoch,
        epoch_size, optimizer,
        initializer, batch_size,
        allow_extra_params
    end

    def self.deserialize(bytes : Slice[UInt8], epoch : Int32,
                         ctx : Array(Context) = [Context.cpu],
                         num_epoch : Int32 = -1,
                         epoch_size : Int32 = -1,
                         optimizer : Optimizer = SGD.new,
                         initializer : Initializer = Uniform.new(0.01_f32),
                         batch_size : Int32 = 128,
                         allow_extra_params : Bool = false) : FeedForward
      symbol, arg_params, aux_params = Model.deserialize bytes
      FeedForward.new symbol, ctx,
        arg_params, aux_params,
        epoch, num_epoch,
        epoch_size, optimizer,
        initializer, batch_size,
        allow_extra_params
    end

    def self.new_builder(model_def : Symbol) : Builder
      Builder.new model_def, nil
    end

    def self.new_builder(sym_gen : SymbolGenerator) : Builder
      Builder.new nil, sym_gen
    end

    class Builder
      @ctx : Array(Context)
      @num_epoch : Int32
      @epoch_size : Int32
      @optimizer : Optimizer
      @initializer : Initializer
      @batch_size : Int32
      @arg_params : Hash(String, NDArray)?
      @aux_params : Hash(String, NDArray)?
      @allowExtraParams : Bool
      @begin_epoch : Int32
      @train_data : DataIter?
      @eval_data : DataIter?
      @eval_metric : EvalMetric
      @kv_store_inst : KVStore?
      @kv_store_type : KVStoreType
      @epoch_end_callback : EpochEndCallback?
      @batch_end_callback : BatchEndCallback?
      @logger : Logger
      @work_load_list : Array(Float32)?
      setter :ctx, :num_epoch, :epoch_size, :optimizer, :initializer, :batch_size
      setter :arg_params, :aux_params, :allow_extra_params, :begin_epoch, :train_data
      setter :eval_data, :eval_metric, :kv_store, :kv_store_type, :epoch_end_callback
      setter :batch_end_callback, :logger, :work_load_list

      def initialize(@model_def : Symbol?, @sym_gen : SymbolGenerator?)
        @ctx = [Context.cpu]
        @num_epoch = -1
        @epoch_size = -1
        @optimizer = SGD.new
        @allow_extra_params = false
        @begin_epoch = 0
        @eval_metric = Accuracy.new
        @kv_store_type = KVStoreType::LOCAL
        @logger = FeedForward.logger
      end

      def build : FeedForward
        raise "Training data missing" if @train_data.nil?
        model = FeedForward.new @model_def, @sym_gen, @ctx, @num_epoch, @epoch_size,
          @optimizer, @initializer, @batch_size, @arg_params, @aux_params, @allow_extra_params, @begin_epoch
        if @kv_store_inst.nil?
          model.fit(@train_data, @eval_data, @eval_metric, @kv_store_type, @epoch_end_callback, @batch_end_callback, @logger, @work_load_list)
        else
          model.fit(@train_data, @eval_data, @eval_metric, @kv_store_inst, @epoch_end_callback, @batch_end_callback, @logger, @work_load_list)
        end
        model
      end

      def setup : FeedForward
        FeedForward.new @model_def, @sym_gen, @ctx, @num_epoch, @epoch_size,
          @optimizer, @initializer, @batch_size, @arg_params, @aux_params, @allow_extra_params, @begin_epoch
      end
    end
  end
end
