module MXNet
  struct Shape
    include Enumerate(MXUInt)
    @shape : Array(MXUInt)
    getter :shape

    def initialize(@shape)
    end

    def [](dim : Int32)
      @shape[dim]
    end

    def [](start : Int, count : Int)
      Shape.new @shape[start, count]
    end

    def [](range : Range(Int, Int))
      Shape.new @shape[range]
    end

    def prod
      @shape.reduce(1) { |acc, i| acc * i }
    end

    def +(other : Shape)
      Shape.new (@shape + other.shape)
    end

    def to_s
      "(#{@shape.join(",")})"
    end

    def_equals_and_hash @shape
    delegate :each, to: @shape
    delegate :size, to: @shape
  end
end
