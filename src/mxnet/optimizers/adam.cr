module MXNet
  abstract class Optimizer
    OPTIMIZER_MAPPING["Adam"] = Adam

    class Adam < Optimizer
      @learning_rate : Float32
      @beta1 : Float32
      @beta2 : Float32
      @epsilon : Float32
      @decay_factor : Float32
      @wd : Float32
      @clip_gradient : Float32
      @lr_scheduler : LRScheduler?
      @time : Int32
      @time_first_index : Int32?

      Adam_JSON = Optimizer_JSON + {
        learning_rate:    Float32,
        beta1:            Float32,
        beta2:            Float32,
        epsilon:          Float32,
        decay_factor:     Float32,
        wd:               Float32,
        clip_gradient:    Float32,
        lr_scheduler:     {nilable: true, type: LRScheduler},
        time:             Int32,
        time_first_index: {nilable: true, type: Int32},
      }
      JSON.mapping(Adam_JSON)

      def initialize(@learning_rate = 0.002_f32,
                     @beta1 = 0.9_f32,
                     @beta2 = 0.999_f32,
                     @epsilon = 1e-8_f32,
                     @decay_factor = (1 - 1e-8_f32),
                     @wd = 0.0_f32,
                     @clip_gradient = 0.0_f32,
                     @lr_scheduler = nil)
        @time = 0
        unless @lr_scheduler.nil?
          @lr_scheduler.base_lr = @learning_rate
        end
      end

      def update(index : Int32, weight : NDArray, grad : NDArray, state)
        lr = if !@lr_scheduler.nil?
               schedule_lr = lr_scheduler[@num_update]
               update_count index
               schedule_lr
             else
               @learning_rate
             end
        lr *= @lr_scale.fetch(index, 1_f32)
        mean, variance = state.as({NDArray, NDArray})
        if @time_first_index.nil?
          @time_first_index = index
          @time = 0
        else
          if @time_first_index.as(Int32) == index
            @time += 1
          end
        end
        t1 = time + 1
        learning_rate = (lr * (1.0 - beta2**t1).sqrt / (1.0 - beta1**t1)).to_f32
        beta1t = beta1 * (@decay_factor**(t1 - 1)).to_f32
        resd_grad = grad * @rescale_grad
        if @clip_gradient != 0.0_f32
          resd_grad = NDArray.clip(resd_grad, -@clip_gradient, @clip_gradient)
        end
        mean_t = (beta1t * mean + (1.0 - beta1t) * resd_grad)
        variance_t = (beta2 * variance + (1.0_f32 - beta2) * resd_grad * resd_grad)
        step = (learning_rate * mean_t / variance_t.sqrt + @epsilon)
        wd = get_wd(index, @wd)
        if wd > 0.0_f32
          step_delta = lr * wd * weight
          step += step_delta
        end
        weight -= step
        mean.set(mean_t)
        variance.set(variance_t)
      end

      def create_state(index : Int32, weight : NDArray)
        @time_first_index = nil
        return NDArray.zeros(weight.shape, weight.context), NDArray.zeros(weight.shape, weight.context)
      end
    end
  end
end
