module MXNet
  class Random
    def self.uniform(low : Float32,
                     high : Float32,
                     shape : Shape? = nil,
                     ctx : Context? = nil,
                     out out_ : NDArray? = nil) : NDArray
      out_copy = if out_.nil?
                   raise MXError.new "shape is required when out is not specified" if shape.nil?
                   empty(shape, ctx)
                 else
                   raise MXError.new "shape and ctx is not needed when out is specified" unless shape.nil? && ctx.nil?
                   out_
                 end
      random_uniform(low, high, out_copy)
    end

    def self.normal(loc : Float32,
                    scale : Float32,
                    shape : Shape? = nil,
                    ctx : Context? = nil,
                    out out_ : NDArray? = nil) : NDArray
      out_copy = if out_.nil?
                   raise MXError.new "shape is required when out is not specified" if shape.nil?
                   empty(shape, ctx)
                 else
                   raise MXError.new "shape and ctx is not needed when out is specified" unless shape.nil? && ctx.nil?
                   out_
                 end
      random_gaussian(loc, scale, out_copy)
    end

    def self.seed(seed_state : Int32)
      MXNet.check_call LibMXNet.mx_random_seed(seed_state)
    end
  end
end
