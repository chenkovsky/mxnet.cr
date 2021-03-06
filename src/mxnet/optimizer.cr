module MXNet
  abstract class Optimizer
    abstract class MXKVStoreUpdater
      abstract def update(key : Int32, recv : NDArray, local : NDArray)
    end

    OPTIMIZER_MAPPING = {} of String => Optimizer

    Optimizer_JSON = {
      lr_scale:           Hash(Int32, Float32),
      num_update:         Int32,
      index_update_count: Hash(Int32, Int32),
      specialized:        Bool,
      weight_set:         Set(Int32),
      rescale_grad:       Float32,
      symbol:             {nilable: true, type: Symbol},
      idx2name:           {nilable: true, type: Hash(Int32, String)},
    }

    def serialize(io = IO::Memory.new)
      prev_size = io.size
      IO::ByteFormat::BigEndian.encode(0, io)
      io << self.class.to_s
      io << '\n'
      to_json io
      cur_size = io.size
      io.seek prev_size
      IO::ByteFormat::BigEndian.encode(cur_size - prev_size, io)
      io.seek cur_size
    end

    def self.deserialize(io : IO) : Optimizer
      line = io.gets
      raise MXError.new "deserializing empty bytes" if line.nil?
      OPTIMIZER_MAPPING[line].from_json(io)
    end

    def self.deserialize(buff : UInt8*) : Optimizer
      body_size = IO::ByteFormat::BigEndian.decode(UInt32, Slice.new(buff, sizeof(UInt32)))
      io = IO::Memory.new Slice(UInt8).new(buff, body_size)
      io.seek sizeof(UInt32)
      deserialize(io)
    end

    class OptimizerUpdater < MXKVStoreUpdater
      @optimizer : Optimizer

      def initialize(@optimizer)
        @states = {} of Int32 => State
      end

      def update(index : Int32, grad : NDArray, weight : NDArray)
        state = if states.has_key? index
                  states[index]
                else
                  new_state = @optimizer.create_state(index, weight)
                  states[index] = new_state
                  new_state
                end
        @optimizer.update(index, weight, grad, state)
      end
    end

    def updater : MXKVStoreUpdater
      OptimizerUpdater.new self
    end

    @lr_scale : Hash(Int32, Float32)
    @num_update : Int32
    @index_update_count : Hash(Int32, Int32)
    @specialized : Bool
    @weight_set : Set(Int32)
    @rescale_grad : Float32
    @symbol : Symbol?
    @idx2name : Hash(Int32, String)?

    def initialize
      @lr_scale = {} of Int32 => Float32
      @num_update = 0
      @index_update_count = {} of Int32 => Int32
      @specialized = false
      @weight_set = Set(Int32).new
      @rescale_grad = 1
      @symbol = nil
      @idx2name = nil
    end

    abstract def update(index : Int32, weight : NDArray, grad : NDArray, state : State)
    abstract def create_state(index : Int32, weight : NDArray) : State
    abstract def finalize_state(state : State)

    def lr_scale=(lr_scale : Hash(Int32, Float32))
      @lr_scale = lr_scale.clone
    end

    def arg_names=(arg_names : Array(String)?)
      unless arg_names.nil?
        @specialized = true
        index = 0
        arg_names.each do |name|
          if !name.ends_with?("data") && !name.ends_with?("label")
            if name.ends_with?("weight")
              @weight_set << index
            end
            index += 1
          end
        end
      end
    end

    setter :rescale_grad, :symbol, :idx2name
    protected def update_count(index : Int32)
      count = @index_update_count.fetch(index, 0) + 1
      @index_update_count[index] = count
      @num_update = {count, num_update}.max
    end
    protected def wd(index : Int32, wd : Float32) : Float32
      if @specialized
        if @weight_set.includes? index
          wd
        else
          0_f32
        end
      else
        wd
      end
    end
  end
end
