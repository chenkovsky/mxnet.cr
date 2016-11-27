module MXNet
  module Callback
    # Callback functions that can be used to track various status during epoch.
    extend self
    @@logger = Logger.new STDOUT

    # Callback to checkpoint the model to prefix every epoch.
    # Parameters
    # ----------
    # prefix : str
    #     The file prefix to checkpoint to
    # period : int
    #     How many epochs to wait before checkpointing. Default is 1.
    # Returns
    def do_checkpoint(prefix : String, period : In32 = 1) : (Int32, Symbol, Hash(String, NDArray), Hash(String, NDArray)) -> Void
      period = {period, 1}.max
      _callback = ->(iter_no : Int32, sym : Symbol, arg : Hash(String, NDArray), aux : Hash(String, NDArray)) {
        if (iter_no + 1) % period == 0
          Model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
        end
      }
      return _callback
    end

    alias BatchEndParam = {epoch: Int32, nbatch: Int32, eval_metric: Metric::EvalMetric?}

    def log_train_metric(period : Int32, auto_reset : Bool = false)
      # Callback to log the training evaluation result every period.
      #     Parameters
      #     ----------
      #     period : int
      #         The number of batch to log the training evaluation metric.
      #     auto_reset : bool
      #         Reset the metric after each log
      #     Returns
      #     -------
      #     callback : function
      #         The callback function that can be passed as iter_epoch_callback to fit.
      _callback = ->(param : BatchEndParam) {
        if param.nbatch % period == 0 && !param.eval_metric.nil?
          metric = param.eval_metric.as(Metric::EvalMetric)
          name_val = metric.name_value
          name_value.each do |name, value|
            @@logger.info("Iter[#{param.epoch}] Batch[#{param.nbatch}] Train-#{name}=#{value}")
          end
          if auto_reset
            metric.reset
          end
        end
      }
      return _callback
    end

    class Speedometer
      # Calculate training speed in frequent

      # Parameters
      # ----------
      # batch_size: int
      #     batch_size of data
      # frequent: int
      #     calculation frequent
      @batch_size : Int32
      @frequent : Int32

      def initialize(@batch_size, @frequent = 50)
        @init = false
        @tic = 0
        @last_count = 0
      end

      def call(param : BatchEndParam)
        count = param.nbatch
        if @last_count > count
          @init = false
        end
        @last_count = count
        if @init
          if count % @frequent == 0
            speed = @frequent * @batch_size / (Time.time - @tic)
            if !param.eval_metric.nil?
              metric = param.eval_metric.as(Metric::EvalMetric)
              name_value = metric.name_value
              metric.reset
              name_value.each do |name, value|
                @@logger.info "Epoch[#{param.epoch}] Batch [#{param.count}]\tSpeed: #{param.speed} samples/sec\tTrain-#{param.name}=#{param.value}"
              end
            else
              @@logger.info "Iter[#{param.epoch}] Batch [#{count}]\tSpeed: #{speed} samples/sec"
            end
            @tic = Time.time
          end
        else
          @init = true
          @tic = Time.time
        end
      end
    end

    class ProgressBar
      @total : Int32
      @bar_len : Int32

      def initialize(@total, @bar_len = 80)
      end

      def call(param : BatchEndParam)
        count = param.nbatch
        filled_len = ((@bar_len * count)/@total.to_f).round
        percents = (100.0*count/total.to_f).ceil
        prog_bar = '=' * filled_len + '-' * (@bar_len - filled_len)
        puts "[#{prog_bar}] #{percents}%\r"
      end
    end
  end
end
