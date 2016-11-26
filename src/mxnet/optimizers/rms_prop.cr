module MXNet
  class RMSProp < Optimizer
    @learning_rate : Float32
    @rescale_gradient : Float32
    @gamma1 : Float32
    @gamma2 : Float32
    @wd : Float32
    @lr_scheduler : LRScheduler?
    @clip_gradient : Float32

    def initialize(@learning_rate = 0.002_f32,
                   @rescale_gradient = 1.0_f32,
                   @gamma1 = 0.95_f32,
                   @gamma2 = 0.9_f32,
                   @wd = 0.0_f32,
                   @lr_scheduler = nil,
                   @clip_gradient = 0_f32)
    end

    def update(index : Int32, weight : NDArray, grad : NDArray, state)
      lr = @learning_rate * @lr_scale.fetch(index, 1_f32)
      n, g, delta = state.as({NDArray, NDArray, NDArray})
      wd = get_wd(index, @wd)
      resd_grad = grad * @rescale_grad
      if @clip_gradient != 0_f32
        resd_grad = NDArray.clip(resd_grad, -@clip_gradient, @clip_gradient)
      end
      n_updated = (1 - @gamma1) * (resd_grad * resd_grad) + @gamma1 * n
      n.set(n_updated)
      g_updated = (1 - @gamma1) * resd_grad + @gamma1 * g
      g.set(g_updated)
      delta_updated = (@gamma2 * delta - lr * (resd_grad / (n - g * g + 1e-4_f32).sqrt + wd * weight))
      delta.set(delta_updated)
      weight += delta
    end

    def create_state(index : Int32, weight : NDArray)
      return NDArray.zeros(weight.shape, weight.context), NDArray.zeros(weight.shape, weight.context), NDArray.zeros(weight.shape, weight.context)
    end
  end
end
