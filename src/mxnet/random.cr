module MXNet
  class Random
    def self.uniform(low : Float32,
                     high : Float32,
                     shape : Shape | Array(Int32) | Nil = nil,
                     ctx : Context? = nil,
                     out out_ : NDArray? = nil) : NDArray
      out_copy = if out_.nil?
                   raise MXError.new "shape is required when out is not specified" if shape.nil?
                   NDArray.empty(shape, ctx)
                 else
                   raise MXError.new "shape and ctx is not needed when out is specified" unless shape.nil? && ctx.nil?
                   out_
                 end
      out_copy.random_uniform(low, high)
    end

    def self.normal(loc : Float32,
                    scale : Float32,
                    shape : Shape | Array(Int32) | Nil = nil,
                    ctx : Context? = nil,
                    out out_ : NDArray? = nil) : NDArray
      out_copy = if out_.nil?
                   raise MXError.new "shape is required when out is not specified" if shape.nil?
                   NDArray.empty(shape, ctx)
                 else
                   raise MXError.new "shape and ctx is not needed when out is specified" unless shape.nil? && ctx.nil?
                   out_
                 end
      out_copy.random_gaussian(loc, scale)
    end

    def self.seed(seed_state : Int32)
      MXNet.check_call LibMXNet.mx_random_seed(seed_state)
    end
  end
end
