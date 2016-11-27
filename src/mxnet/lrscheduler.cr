module MXNet
  abstract class LRScheduler
    def initialize(@base_lr : Float32 = 0.01_f32)
    end

    abstract def apply(num_update : Int32) : Float32
  end

  class FactorScheduler < LRScheduler
    def initialize(@step : Int32, @factor : Float32)
      super
      @count = 0
      raise MXError.new "Schedule step must be greater or equal than 1 round" if @step < 1
      raise MXError.new "Factor must be less than 1 to make lr reduce" if factor >= 1.0
    end

    def apply(num_update : Int32)
      if num_update > @count + @step
        @count += @step
        @base_lr *= @factor
        @@logger.info "Update #{num_update}: Change learning rate to #{@base_lr}"
      end
      @base_lr
    end
  end
end
