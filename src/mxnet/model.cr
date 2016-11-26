require "time"

module MXNet
  module Model
    def self.save_check_point(prefix : String, epoch : Int32, symbol : Symbol,
                              arg_params : Hash(String, NDArray),
                              aux_params : Hash(String, NDArray))
      symbol.save("#{prefix}-symbol.json")
      save_dict = {} of String => NDArray
      arg_params.each { |k, v| save_dict["arg:#{k}"] = v }
      aux_params.each { |k, v| save_dict["aux:#{k}"] = v }
      param_name = "#{prefix}-#{epoch}.params"
      NDArray.save(param_name, save_dict)
    end

    def self.load_check_point(prefix : String, epoch : Int32) : {Symbol, Hash(String, NDArray), Hash(String, NDArray)}
      symbol = Symbol.loads("#{prefix}-symbol.json")
      save_dict = NDArray.load("#{prefix}-#{epoch}.params")
      arg_params = Hash(String, NDArray).new
      aux_params = Hash(String, NDArray).new
      save_dict.each do |k, v|
        case k
        when .start_with? "arg"
          name = k[3...k.size]
          arg_params[name] = v
        when .start_with? "aux"
          name = k[3...k.size]
          arg_params[name] = v
        end
      end
      return symbol, arg_params, aux_params
    end

    protected def create_kv_store(kv_store : String, num_device : Int32, arg_params : Hash(String, NDArray)) : {KVStore?, Bool}
      if num_device == 1 && !kv_store.includes? "dist"
        return nil, false
      else
        kv_type = kv_store
        if kv_type == "local"
          max_size = (arg_params.values.map &.shape.product).max
          kv_type = if max_size < 1024 * 1024 * 16
                      "local_update_cpu"
                    else
                      "local_allreduce_cpu"
                    end
        end
        return KVStore.create(kv_store), !kv_type.includes? "local_allreduce"
      end
    end
    protected def create_kv_store(kv_store : KVStore)
      return kv_store, !kv_store.nil? && !kv_store.type.includes?("local_allreduce")
    end
    protected def initialize_kv_store(kv_store : KVStore,
                                      param_arrays : Array(Array(NDArray)),
                                      arg_params : Hash(String, NDArray),
                                      param_names : Array(String),
                                      update_on_kv_store : Bool)
      raise MXError.new "param_arrays.size != param_names.size" if param_arrays.size != param_names.size
      (0...param_arrays.size).each do |idx|
        param_on_devs = param_arrays[idx]
        kv_store.init(idx, arg_params[param_names[idx]])
        if update_on_kv_store
          kv_store.pull(idx, param_on_devs, -idx)
        end
      end
    end
    protected def update_params_on_kv_store(param_arrays : Array(Array(NDArray)),
                                            grad_arrays : Array(Array(NDArray)?),
                                            kv_store : KVStore?)
      raise MXError.new "param_arrays.size != grad_arrays.size" if param_arrays.size != grad_arrays.size
      (0...param_arrays.size).each do |idx|
        arg_list = param_arrays[idx]
        grad_list = grad_arrays[idx]
        unless grad_lst.nil?
          unless kv_store.nil?
            kv_store.push(idx, grad_list, -idx)
            kv_store.push(idx, arg_list, -idx) # what?
          end
        end
        raise MXError.new "param_arrays[#{idx}].size != grad_arrays[#{idx}].size" if arg_list.size != grad_list.size
        (0...arg_list.size).each do |idx2|
          w = arg_list[idx2]
          g = grad_list[idx2]
          updater.update(idx * num_device + idx2, g, w)
        end
      end
    end
    protected def update_params(param_arrays : Array(Array(NDArray)),
                                grad_arrays : Array(Array(NDArray)?),
                                updater : MXKVStoreUpdater,
                                num_device : Int32,
                                kv_store : KVStore? = nil)
      raise MXError.new "param_arrays.size != grad_arrays.size" if param_arrays.size != grad_arrays.size
      (0...param_arrays.size).each do |idx|
        arg_list = param_arrays[idx]
        grad_list = grad_arrays[idx]
        unless grad_list.nil?
          unless kv_store.nil?
            kv_store.push idx, grad_list, -idx
            kv_store.push idx, grad_list, -idx
          end
        end
      end
    end
    protected def train_multi_device(symbol : Symbol,
                                     ctx : Array(Context),
                                     arg_names : Array(String),
                                     param_names : Array(String),
                                     aux_names : Array(String),
                                     arg_params : Hash(String, NDArray),
                                     aux_params : Hash(String, NDArray),
                                     begin_epoch : Int32,
                                     end_epoch : Int32,
                                     epoch_size : Int32,
                                     optimizer : Optimizer,
                                     kv_store : KVStore?,
                                     update_on_kv_store : Bool,
                                     train_data : DataIter,
                                     eval_metric : EvalMetric,
                                     eval_data : DataIter? = nil,
                                     epoch_end_callback : EpochEndCallback? = nil,
                                     batch_end_callabck : BatchEndCallback? = nil,
                                     work_load_list : Array(Float32)? = nil,
                                     monitor : Monitor? = nil,
                                     sym_gen : SymbolGenerator? = nil)
      executor_manager = DataParallelExecutorManager.new(
        symbol: symbol,
        sym_gen: sym_gen,
        ctx: ctx,
        train_data: train_data,
        param_names: param_names,
        arg_names: arg_names,
        aux_names: aux_names,
        work_load_list: work_load_list
      )
      unless monitor.nil?
        executor_manager.install_monitor(monitor)
      end
      executor_manager.params = {arg_params, aux_params}
      updater_local = optimizer.updater
      unless kv_store.nil?
        initialize_kv_store(kv_store, executor_manager.param_arrays, arg_params, executor_manager.param_names, update_on_kv_store)
      end
      if update_on_kv_store
        unless kv_store.nil?
          kv_store.optimizer = optimizer
        end
      end
      (begin_epoch...end_epoch).each do |epoch|
        tic = Time.now
        eval_metric.reset
        n_batch = 0
        epoch_done = false
        train_data.reset
        while !epoch_done
          do_reset = true
          train_data.each do |data_batch|
            executor_manager.load_data_batch data_batch
            unless monitor.nil?
              monitor.tic
            end
            executor_manager.forward(is_train: true)
            executor_manager.backward
            if update_on_kv_store
              update_params_on_kv_store(executor_manager.param_arrays, executor_manager.grad_arrays, kv_store)
            else
              update_params(executor_manager.param_arrays, executor_manager.grad_arrays, updater_local, ctx.size, kv_store)
            end
            unless monitor.nil?
              monitor.toc_print
            end
            executor_manager.update_metric(eval_metric, data_batch.label)
            n_batch += 1
            unless batch_end_callabck.nil?
              batch_end_callabck.call(epoch, n_batch, eval_metric)
            end
            if epoch_size != -1 && n_batch >= epoch_size
              do_reset = false
            end
            break if do_reset
          end
          if do_reset
            train_data.reset
          end
          epoch_done = epoch_size == -1 || n_batch >= epoch_size
        end
        name, value = eval_metric.get
        @@logger.info "Epoch[#{epoch}] Train-#{name}=#{value}"
        toc = Time.now
        @@logger.info "Epoch[#{epoch} Time cost=#{toc - tic}"
        unless eval_data.nil?
          eval_metric.reset
          eval_data.reset
          eval_data.each do |eval_batch|
            executor_manager.load_data_batch eval_batch
            executor_manager.forward is_train: false
            executor_manager.update_metric eval_metric, eval_batch.label
          end
          name, value = eval_metric.get
          @@logger.info "Epoch[#{epoch}] Validation-#{name}=#{value}"
        end
        if !epoch_end_callback.nil? || epoch + 1 == end_epoch
          executor_manager.copy_to(arg_params, aux_params)
        end
        unless epoch_end_callback.nil?
          epoch_end_callback.call(epoch, symbol, arg_params, aux_params)
        end
      end
    end
  end
end
