module MXNet
  class MXNet
    @learning_rate : Float32
    @rescale_gradient : Float32
    @wd : Float32
    @clip_gradient : Float32
    @lr_scheduler : LRScheduler?

    def initialize(@learning_rate = 0.01_f32,
                   @rescale_gradient = 1.0_f32,
                   @wd = 0.0001_f32,
                   @clip_gradient = 0_f32,
                   @lr_scheduler = nil)
      if !@lr_scheduler.nil?
        @lr_scheduler.base_lr = @learning_rate
      end
    end

    def update(index : Int32, weight : NDArray, grad : NDArray, state)
      lr = if @lr_scheduler.nil?
             @learning_rate
           else
             scheduler_lr = @lr_scheduler[@num_update]
             update_count index
             scheduler_lr
           end
      lr *= @lr_scale.fetch(index, 1_f32)
      wd = get_wd(index, @wd)
      resd_grad = grad * @rescale_grad
      if @clip_gradient != 0_f32
        resd_grad = NDArray.clip(resd_grad, -@clip_gradient, @clip_gradient)
      end
      adder = @wd * weight
      adder += resd_grad
      adder *= -(lr / 2)
      norm = Random.normal(0_f32, lr.sqrt.to_f32, weight.shape, weight.context)
      adder += norm
      weight += adder
    end

    def create_state(index : Int32, weight : NDArray)
      nil
    end
  end
end
