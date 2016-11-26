module MXNet
  class AdaDelta < Optimizer
    @rho : Float32
    @rescale_gradient : Float32
    @epsilon : Float32
    @wd : Float32
    @clip_gradient : Float32

    def initialize(@rho = 0.05_f32,
                   @rescale_gradient = 1.0_f32,
                   @epsilon = 1e-8_f32,
                   @wd = 0.0_f32,
                   @clip_gradient = 0_f32)
    end

    def update(index : Int32, weight : NDArray, grad : NDArray, state)
      resd_grad = grad * @rescale_grad
      if @clip_gradient != 0_f32
        resd_grad = NDArray.clip(resd_grad, -@clip_gradient, @clip_gradient)
      end
      acc_g, acc_delta = state.as({NDArray, NDArray})
      new_acc_g = (@rho * acc_g + (1.0_f32 - @rho) * resd_grad * resd_grad)
      acc_g.set(new_acc_g)
      current_delta = ((acc_delta + @epsilon).sqrt / (acc_g + @epsilon).sqrt * resd_grad)
      new_acc_delta = (@rho * acc_delta + (1.0_f32 - @rho) * current_delta * current_delta)
      acc_delta.set(new_acc_delta)
      weight *= (1 - @wd)
      weight -= current_delta
    end

    def create_state(index : Int32, weight : NDArray)
      return NDArray.zeros(weight.shape, weight.context), NDArray.zeros(weight.shape, weight.context)
    end
  end
end
