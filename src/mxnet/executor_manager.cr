require "logger"

module MXNet
  class DataParallelExecutorManager
    @@logger = Logger.new(STDOUT)
    @symbol : Symbol
    @ctx : Array(Context)
    @param_names : Array(String)
    @arg_names : Array(String)
    @aux_names : Array(String)
    @train_data : DataIter
    @work_load_list : Array(Float32)
    @sym_gen : SymbolGenerator?
    @slice : Array({Int32, Int32})
    @param_name_set : Set(String)
    @cur_exec_grp : DataParallelExecutorGroup?

    def num_device
      @ctx.size
    end

    def initialize(@symbol, @ctx, @param_names, @arg_names, @aux_names, @train_data, work_load_list : Array(Float32)? = nil, @sym_gen = nil)
      @@logger.info("Start training with #{@ctx.map { |x| x.to_s }}")
      ExecutorManager.check_arguments(@symbol)
      @work_load_list = if work_load_list.nil?
                          (0...num_device).map { |idx| 1_f32 }
                        else
                          work_load_list
                        end
      raise MXError.new "Invalid settings for work load." unless @work_load_list.size == num_device
      @slices = ExecutorManager.split_input_slice tran_data.batch_size, work_load_list
      @param_name_set = param_names.to_set
      @exec_grp = DataParallelExecutorGroup.new @symbol, @arg_names, @param_name_set, @ctx, @slices, @train_data
      @exec_grp_bucket = Hash(BucketKey, DataParallelExecutorGroup).new
      unless @sym_gen.nil?
        @exec_grp_bucket[@train_data.default_bucket_key, @exec_grp]
      end
    end

    delegate :param_arrays, to: @exec_grp
    delegate :grad_arrays, to: @exec_grp
    delegate :aux_arrays, to: @exec_grp

    def install_monitor(monitor : Monitor)
      raise MXError.new "Monitoring is not implemented for bucketing" unless @sym_gen.nil?
      @exec_grp.train_execs.each do |x|
        monitor.install
      end
    end

    def params=(arg_params : Hash(String, NDArray), aux_params : Hash(String, NDArray))
      @exec_grp.train_execs.each do |e|
        e.copy_params_from(arg_params, aux_params)
      end
    end

    def copy_to(arg_params : Hash(String, NDArray), aux_params : Hash(String, NDArray))
      @param_names.zip(@param_arrays) do |name, block|
        weight = block.map { |x| x.copy_to(Context.cpu).reduce(0) { |acc, i| acc + i } } / block.size
        weight.copy_to arg_params[name]
      end
      @aux_names.zip(@aux_arrays) do |name, block|
        weight = block.map { |x| x.copy_to(Context.cpu).reduce(0) { |acc, i| acc + i } } / block.size
        weight.copy_to aux_params[name]
      end
    end

    def load_data_batch(data_batch : DataBatch)
      @cur_exec_grp = if @sym_gen.nil?
                        key = data_batch.default_bucket_key
                        if !@exec_grp_bucket.has_key? key
                          sym = @sym_gen.generate(key)
                          grp = DataParallelExecutorGroup.new sym, @arg_names, @param_name_set,
                            @ctx, @slices, @data_batch, @shared_group = exec_grp
                          @exec_grp_bucket[key] = grp
                        end
                        @exec_grp_bucket[key]
                      else
                        @exec_grp
                      end
      @cur_exec_grp.load_data_batch data_batch
    end

    def forward(is_train : Bool = false)
      @cur_exec_grp.as(DataParallelExecutorGroup).forward(is_train: is_train)
    end

    def backward
      @cur_exec_grp.as(DataParallelExecutorGroup).backward
    end

    def update_metric(metric : EvalMetric, labels : Array(NDArray))
      @cur_exec_grp.update_metric(metric, labels)
    end
  end

  class ExecutorManager
    # Get input slice from the input shape.
    # @param batchSize The number of samples in a mini-batch.
    # @param workLoadList The list of work load for different devices, in the same order as ctx
    # @return The split slices to get a specific slice.
    # @throws IllegalArgumentException
    #         If there are two many splits such that some slice can be empty.
    protected def split_input_slice(batch_size : Int32, work_load_list : Array(Float32)) : Array({Int32, Int32})
      total_work_load = work_load_list.sum
      batch_num_list = work_load_list.map { |work_load| (work_load * batch_size / total_work_load).round }
      batch_num_sum = batch_num_list.sum
      if batch_num_sum < batch_size
        batch_num_list[-1] += batch_size - batch_num_sum
      end
      end_ = 0
      batch_num_list.map do |batch_num|
        begin_ = {end_, batch_size}.min
        end_ = {begin_ + batch_num, batch_size}.min
        raise MXError.new "Too many slices such that some splits are empty" unless begin_ < end_
        {begin_, end_}
      end
    end

    # Check the argument names of symbol.
    # This function checks the duplication of arguments in Symbol.
    # The check is done for feedforward net for now.
    # @param symbol The network configuration
    protected def check_arguments(symbol : Symbol)
      arg_names = symbol.arguments
      raise MXError.new ("Find duplicated argument name," +
                         "please make the weight name non-duplicated(using name arguments)," +
                         "arguments are #{arg_names}"
        ) unless arg_names.to_set.size == arg_names.size
      aux_names = symbol.auxiliary_states
      raise MXError.new ("Find duplicated auxiliary param name," +
                         "please make the weight name non-duplicated(using name arguments)," +
                         "arguments are #{aux_names}") unless aux_names.to_set.size == aux_names.size
    end

    # Load a list of arrays into a list of arrays
    protected def load_general(data : Array(NDArray), targets : Array(NDArray))
      data.zip(targets) do |d_src, d_target|
        raise MXError.new "src shape #{d_src.shape} mismatch dst shape #{d_target.shape}" unless d_src.shape == d_target.shape
        d_src.copy_to d_target
      end
    end
    protected def load_general_multi(data : Array(NDArray), targets : Array(Array({Int32, Int32, NDArray})))
      data.zip(targets).each do |src, d_targets|
        d_targets.each do |start_, end_, dst|
          sliced = src[start_, end_]
          raise MXError.new "src shape #{sliced.shape} mismatch dst shape #{dst.shape}" unless sliced.shape == dst.shape
          sliced.copy_to dst
        end
      end
    end
    protected def load_data_multi(batch : DataBatch, targets : Array(Array({Int32, Int32, NDArray})))
      load_general_multi(batch.data, targets)
    end
    protected def load_data(batch : DataBatch, targets : Array(NDArray))
      load_general(batch.data, targets)
    end
    protected def load_label_multi(batch : DataBatch, targets : Array(Array(Int32, Int32, NDArray)))
      load_general(batch.label, targets)
    end
    protected def load_label(batch : DataBatch, targets : Array(NDArray))
      load_general(batch.label, targets)
    end
    protected def bind_exec(sym : Symbol, ctx : Context, input_shapes : Hash(String, Shape),
                            param_names : Set(String), need_grad : Bool = false,
                            grads : Set(String)? = nil, base_exec : Executor? = nil,
                            shared_data_arrays : Hash(String, NDArray)? = nil,
                            input_types : Hash(String, MXType)? = nil)
      arg_shape, _, aux_shape = sym.infer_shape input_shapes
      raise MXError.new "arg_shape is nil" if arg_shape.nil?
      input_types_update = if input_types.nil?
                             input_shapes.map { |key, _| {key, MXType::Float32_T} }.to_h
                           else
                             input_types
                           end
      arg_types, _, aux_types = sym.infer_type input_types_update
      raise MXError.new "arg_types is nil" if arg_types.nil?
      grad_arrays = if need_grad
                      Hash(String, NDArray).new
                    else
                      nil
                    end
      arg_names = sym.arguments
      grad_set = if !need_grad
                   Set(String).new
                 elsif grads.nil?
                   arg_names.to_set - input_shapes.keys
                 else
                   grads
                 end
      grad_req = arg_names.map { |name| grad_set.includes?(name) ? SymbolFunction::BindReq::WRITE : SymbolFunction::BindReq::NULL }
      arg_arrays = arg_names.map_with_index do |name, i|
        if !param_names.includes? name
          if !shared_data_arrays.nil? && shared_data_arrays.has_key? name
            arr = shared_data_arrays[name]
            if arr.shape.prod >= arg_shape[i].prod
              arr.reshape arg_shape[i]
            else
              @@logger.warn("bucketing: data #{name} has a shape #{arg_shape[i]}," +
                            "which is larger than already allocated shape #{arr.shape}." +
                            "Need to re-allocate. Consider putting default_bucket_key " +
                            "to be the bucket taking the largest input for better memory sharing."
              )
              zeros = NDArray.zeros(arg_shape[i], ctx)
              shared_data_arrays[name] = zeros
              zeros
            end
          else
            zeros = NDArray.zeros arg_shape[i], ctx
            unless shared_data_arrays.nil?
              shared_data_arrays[name] = zeros
            end
            zeros
          end
        else
          if base_exec.nil?
            if grad_set.includes? name
              grad_arr = NDArray.zeros arg_shape[i], ctx
              grad_arrays[name] = grad_arr
            end
            NDArray.zeros arg_shape[i], ctx
          else
            arr = base_exec.arg_dict[name]
            raise MXError.new "arr.shape == arg_shape[#{idx}]" unless arr.shape == arg_shape[i]
            if grad_set.includes? name
              grad_arrays[name] = base_exec.grad_dict[name]
            end
            arr
          end
        end
      end
      aux_arrays = if base_exec.nil?
                     aux_shape.zip(aux_types).map do |s, t|
                       NDArray.zeros s, ctx
                     end
                   else
                     base_exec.aux_arrays.each do |a, i|
                       raise MXError.new "aux_shape[i] == a.shape" unless aux_shape[i] == a.shape
                     end
                     base_exec.aux_arrays.to_a
                   end
      sym.bind ctx, arg_arrays, grad_arrays, grad_req, aux_arrays, group2ctx: nil, shared_exec: base_exec
    end
  end

  class DataParallelExecutorGroup
    @sym : Symbol
    @arg_names : Array(String)
    @param_names : Set(String)
    @ctx : Array(Context)
    @slices : Array({Int32, Int32})
    @provided_data : Hash(String, Shape)
    @provided_label : Hash(String, Shape)
    @shared_group : DataParallelExecutorGroup?
    @data_names : Array(String)
    @label_names : Array(String)
    @aux_names : Array(String)
    @param_idx : Array(Int32)
    @param_names_comb : Set(String)
    @train_execs : Array(Executor)
    @data_arrays : Array({Int32, Int32, NDArray})
    @label_arrays : Array({Int32, Int32, NDArray})
    @param_arrays : Array(NDArray)
    @grad_arrays : Array(NDArray)
    @aux_arrays : Array(NDArray)

    def initialize(@sym, @arg_names, @param_names, @ctx, @slices, @provided_data, @provided_label, @shared_group)
      ExecutorManager.check_arguments sym
      shared_data_arrays = if @shared_group.nil?
                             ctx.map { |e| Hash(String, NDArray).new }
                           else
                             @shared_group.as(DataParallelExecutorGroup).shared_data_arrays
                           end
      @data_names = @provided_data.map { |k, _| k }
      @label_names = @provided_label.map { |k, _| k }
      @aux_names = sym.auxiliary_states
      @param_idx = @arg_names.map_with_index { |name_i| name_i }.select! { |name, i| @param_names.includes? name }.map { |name, i| i }
      @param_names_comb = param_idx.map { |i| @arg_names[i] }.to_set
      @train_execs = ctx.map_with_index do |ctxi, i|
        data_shapes = (provided_data + provided_label).map { |name, shape|
          {name, (Shape.new(@slices[i][1] - @slices[i][0]) + shape.slice(1, shape.size))}
        }.to_h
        shared_exec = shared_group.nil? ? nil : shared_group.train_execs[i]
        ExecutorManager.bind_exec(@sym, ctxi, data_shapes, @param_names_comb,
          need_grad: true, base_exec: shared_exec, shared_data_arrays: shared_data_arrays[i])
      end
      @data_arrays = @data_names.map { |name| @train_execs.map_with_index do |e, i|
        {@slices[i][0], @slices[i][1], e.arg_dict[name]}
      end }
      @label_arrays = @label_names.map { |name| @train_execs.map_with_index do |e, i|
        {@slices[i][0], @slices[i][1], e.arg_dict[name]}
      end }
      @param_arrays = @param_idx.map { |i| @train_execs.map { |e| e.arg_arrays[i] } }
      @grad_arrays = @param_idx.map { |i| @train_execs.map { |e| e.grad_arrays[i] } }
      @aux_arrays = (0...@aux_names.size).map { |i| @train_execs.map { |e| e.aux_arrays[i] } }
    end

    def initialize(sym, arg_names, param_names, ctx, slices, train_data, shared_group = nil)
      initialize(sym, arg_names, param_names, ctx, slices, train_data.provided_data, train_data.provided_label, shared_group)
    end

    def load_data_batch(data_batch : DataBatch)
      ExecutorManager.load_data_multi(data_batch, @data_arrays)
      ExecutorManager.load_label_multi(data_batch, @label_arrays)
    end

    def forward(is_train : Bool = false)
      @train_execs.each do |e|
        e.forward(is_train: is_train)
      end
    end

    def backward
      @train_execs.each do |e|
        e.backward
      end
    end

    def update_metric(metric : EvalMetric, labels : Array(NDArray))
      @train_execs.zip(@slices) do |texec, islice|
        labels_slice = labels.map { |l| l.slice(islice) }
        metric.update(labels_slice, texec.outputs)
      end
    end
  end
end
