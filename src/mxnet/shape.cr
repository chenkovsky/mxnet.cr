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

    def [](from : Int32, end to : Int32)
      Shape.new @shape[from, to - from]
    end

    def drop(dim : Int32)
      shape = @shape.clone
      shape.delete_at dim
      Shape.new shape
    end

    def head
      @shape[0]
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
