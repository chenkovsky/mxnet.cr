module MXNet
  abstract class Initializer
    enum Method
      UPSAMPLING
      BIAS
      GAMMA
      BETA
      WEIGHT
      MOVING_MEAN
      MOVING_VAR
      MOVING_AVG
    end

    def apply(method : Method, arr : NDArray)
      case method
      when UPSAMPLING
        init_bilinear(method, arr)
      when BIAS
        init_bias(method, arr)
      when GAMMA
        init_gamma(method, arr)
      when BETA
        init_beta(method, arr)
      when WEIGHT
        init_weight(method, arr)
      when MOVING_MEAN
        init_zero(method, arr)
      when MOVING_VAR
        init_zero(method, arr)
      when MOVING_AVG
        init_zero(method, arr)
      end
    end

    protected def init_bilinear(method : Method, arr : NDArray)
      weight = Array(Float32).new(size: arr.size, value: 0.0_f32)
      shape = arr.shape
      f = shape[3] / 2.0_f32
      c = (2 * f - 1 - f % 2) / (2.0_f32 * f)
      (0...arr.size).each do |i|
        x = i % shape[3]
        y = (i / shape[3]) % shape[2]
        weight[i] = (1 - Math.abs(x / f - c)) * (1 - Math.abs(y / f - c))
      end
      arr.set(NDArray.array(weight, shape))
    end
    protected def init_zero(method : Method, arr : NDArray)
      arr.set(0_f32)
    end
    protected def init_bias(method : Method, arr : NDArray)
      arr.set(0_f32)
    end
    protected def init_gamma(method : Method, arr : NDArray)
      arr.set(0_f32)
    end
    protected def init_beta(method : Method, arr : NDArray)
      arr.set(0_f32)
    end
    protected abstract def init_weight(method : Method, arr : NDArray)

    protected def init_default(method : Method, arr : NDArray)
      raise MXError.new "Unknown initialization pattern for #{method}."
    end

    class Uniform < Initializer
      def initialize(@scale : Float = 0.07_f32)
      end

      def init_weight(method : Method, arr : NDArray)
        Random.uniform(-@scale, @scale, out: arr)
      end
    end

    class Normal < Initializer
      def initialize(@sigma : Float = 0.01_f32)
      end

      def init_weight(method : Method, arr : NDArray)
        Random.normal(0, @sigma, out: arr)
      end
    end

    class Xavier < Initializer
      enum RndType
        GAUSSIAN
        UNIFORM
      end
      enum FactorType
        AVG
        IN
        OUT
      end

      # Initialize the weight with Xavier or similar initialization scheme.
      #
      # @param rndType Options are: "gaussian" or "uniform"
      # @param factorType Options are: "avg", "in", "out"
      # @param magnitude scale of random number range
      def initialize(@rnd_type : RndType, @factor_type : FactorType, @magnitude : Float32 = 3)
      end

      def init_weight(method : Method, arr : NDArray)
        shape = arr.shape
        fan_in = shape[1, shape.size].prod
        fan_out = shape[0]
        factor = 1_f32
        factor = case @factor_type
                 when FactorType::AVG
                   (fan_in + fan_out) / 2_f32
                 when FactorType::IN
                   fan_in
                 when FactorType::OUT
                   fan_out
                 end
        scale = Math.sqrt(@magnitude / factor).to_f32
        case @rnd_type
        when RndType::UNIFORM
          Random.uniform(-scale, scale, out: arr)
        when RndType::GAUSSIAN
          Random.normal(0, scale, out: arr)
        end
      end
    end
  end
end
