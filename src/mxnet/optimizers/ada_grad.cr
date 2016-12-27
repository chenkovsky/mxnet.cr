module MXNet
  abstract class Optimizer
    OPTIMIZER_MAPPING["AdaGrad"] = AdaGrad

    class AdaGrad < Optimizer
      @learning_rate : Float32
      @rescale_gradient : Float32
      @epsilon : Float32
      @wd : Float32

      AdaGrad_JSON = Optimizer_JSON + {
        learning_rate:    Float32,
        rescale_gradient: Float32,
        epsilon:          Float32,
        wd:               Float32,
        clip_gradient:    Float32,
      }
      JSON.mapping(AdaGrad_JSON)

      def initialize(@learning_rate = 0.05_f32, @rescale_gradient = 1.0_f32,
                     @epsilon = 1e-7_f32, @wd = 0.0_f32)
      end

      def update(index : Int32, weight : NDArray, grad : NDArray, state)
        lr = @learning_rate
        resd_grad = @rescale_gradient * grad
        history = state.as(NDArray)
        grad_squared = resd_grad * resd_grad
        history += grad_squared
        new_weight = (-lr * (resd_grad / (history + @epsilon).sqrt) + @wd * weight)
        weight += new_weight
      end

      def create_state(index : Int32, weight : NDArray) : NDArray
        NDArray.zeros(weight.shape, weight.context)
      end
    end
  end
end
