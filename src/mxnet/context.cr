module MXNet
  class Context
    enum DeviceType
      CPU        = 1
      GPU        = 2
      CPU_PINNED = 3
    end
    @device_type : DeviceType
    @device_id : Int32
    getter :device_type, :device_id
    @@default_ctx : Context = Context.new DeviceType::CPU, 0

    def initialize(@device_type, @device_id = 0)
    end

    def self.with(ctx)
      old_ctx = @@default_ctx
      @@default_ctx = ctx
      yield ctx
      @@default_ctx = old_ctx
    end

    def to_s(io)
      io << "#{@device_type.to_s}(#{@device_id})"
    end

    def inspect(io)
      to_s io
    end

    def self.default_ctx
      @@default_ctx
    end

    def self.cpu(device_id = 0)
      Context.new DeviceType::CPU, device_id
    end

    def self.gpu(device_id = 0)
      Context.new DeviceType::GPU, device_id
    end

    def self.current_context
      default_context
    end

    def_equals(@device_type, @device_id)
  end
end
