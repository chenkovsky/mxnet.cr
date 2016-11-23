module MXNet
  class Monitor
    @interval : Int32
    @stat_func : NDArray -> NDArray
    @activated : Bool
    @queue : Deque({Int32, String, NDArray})
    @exes : Deque(Executor)

    def initialize(@interval, stat_func : (NDArray -> NDArray)?)
      if stat_func.nil?
        @stat_func = ->(x : NDArray) {
          NDArray.norm(x) / Math.sqrt(x.size.to_f64).to_f32
        }
      else
        @stat_func = stat_func
      end
      @activated = false
      @queue = Deque({Int32, String, NDArray}).new
      @step = 0
      @exes = Deque(Executor).new
      @stat_helper = ->(name : String, arr : NDArrayHandle) {
        if activated
          array = NDArray.new(arr, writable: false)
          queue << ({@step, name, @stat_func.call(array)})
        end
      }
    end

    def install(exe : Executor)
      exe.monitor_callback = @stat_helper
      @exes << exe
    end

    def tic
      if @step % @interval == 0
        exes.each do |exe|
          exe.arg_arrays.each &.wait_to_read
        end
        @queue = Deque({Int32, String, NDArray}).new
        @activated = true
      end
      @step += 1
    end

    def toc
      if @activated
        @exes.each do |exe|
          exe.arg_arrays.each &.wait_to_read
        end
        @exes.each do |exe|
          exe.symbol.arguments.zip(exe.arg_arrays) do |name, array|
            @queue << ({@step, name, @stat_func.call(array)})
          end
        end
        @activated = false
        res = Deque({Int32, String, String}).new
        @queue.each do |n, k, v|
          if v.shape == Shape.new([1])
            res << ({n, k, v.to_scalar.to_s})
          else
            res << ({n, k, "#{v.to_a.join(",")}"})
          end
        end
        @queue = Deque({Int32, String, NDArray}).new
        res
      else
        Deque({Int32, String, String}).new
      end
    end

    def puts_toc
      res = toc
      res.each do |n, k, v|
        @@logger.info "Batch: #{n} #{k} #{v}"
      end
    end
  end
end
