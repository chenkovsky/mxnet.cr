module MXNet
  class KVStore
    enum Type
      Local
      Local_Allreduce_Device
      Dist_Sync
      Dist_Async
      Dist

      def to_s
        case self
        when Local
          "local"
        when Local_Allreduce_Device
          "local_allreduce_device"
        when Dist_Sync
          "dist_sync"
        when Dist_Async
          "dist_async"
        else
          # when Dist
          "dist"
        end
      end
    end
    @[Flags]
    enum GroupNode
      Scheduler
      Server
      Worker
    end

    def self.create(type t : Type = Type::Local)
      MXNet.check_call LibMXNet.mx_kv_store_create(t.to_s, out handle)
      KVStore.new handle
    end

    @handle : LibMXNet::KVStoreHandle
    @updater_func : Optimizer::MXKVStoreUpdater?

    def initialize(@handle)
    end

    def init(key : Array(Int32), values : Array(NDArray))
      raise MXError.new "len(keys) != len(values)" if keys.size != values.size
      value_ptrs = values.map &.handle
      MXNet.check_call LibMXNet.mx_kv_store_init(@handle, keys.size, keys, value_ptrs)
    end

    def init(key : Int32, value : NDArray)
      init([key], [value])
    end

    def push(keys : Array(Int32), values : Array(NDArray), priority : Int32 = 0)
      raise MXError.new "len(keys) != len(values)" if keys.size != values.size
      value_ptrs = values.map &.handle
      MXNet.check_call LibMXNet.mx_kv_store_push(@handle, keys.size, keys, value_ptrs, priority)
    end

    def push(key : Int32, value : Array(NDArray) | NDArray, priority : Int32 = 0)
      key_, value_ = if value.is_a? NDArray
                       {[key], [value]}
                     else
                       {Array.new(size: value.size, value: key), value}
                     end
      push key_, value_, priority
    end

    def pull(keys : Array(Int32), outs : Array(NDArray), priority : Int32 = 0)
      raise MXError.new "len(keys) != len(outs)" if keys.size != outs.size
      out_ptrs = outs.map &.handle
      MXNet.check_call LibMXNet.mx_kv_store_pull @handle, keys.size, keys, out_ptrs, priority
    end

    def pull(key : Int32, outs : Array(NDArray) | NDArray, priority : Int32 = 0)
      key_, outs_ = if outs.is_a? NDArray
                      {[key], [outs]}
                    else
                      {Array.new(size: outs.size, value: key), outs}
                    end
      pull key_, outs_, priority
    end

    def type
      kv_type = Pointer(UInt8).null
      MXNet.check_call LibMXNet.mx_kv_store_get_type(@handle, out kv_type)
      String.new kv_type
    end

    def num_workers
      size = 0
      MXNet.check_call LibMXNet.mx_kv_store_get_group_size(@handle, out size)
      size
    end

    def rank
      rank = 0
      MXNet.check_call LibMXNet.mx_kv_store_get_group_size(@handle, out rank)
      rank
    end

    def optimizer=(optimizer : Optimizer)
      is_worker = 0
      MXNet.check_call LibMXNet.mx_kv_store_is_worker_node(out is_worker)
      if self.type.includes?("dist") && is_worker != 0
        opt_serialized = Serializer.serializer.serialize(optimizer)
        cmd = Serializer.encode_base64_string opt_serialized
        @@logger.debug "Send optimizer to server: #{cmd}"
        send_command_to_servers 0, cmd
      else
        self.updater = optimizer.updater
      end
    end

    def updater=(updater : MXKVStoreUpdater)
      @updater_func = updater
      MXNet.check_call LibMXNet.mx_kv_store_set_updater(@handle, @updater_func)
    end

    def barrier
      MXNet.check_call LibMXNet.mx_kv_store_barrier(@handle)
    end

    def num_dead_node(node_id : Int32)
      num = 0
      MXNet.check_call LibMXNet.mx_kv_store_get_num_dead_node(@handle, node_id, out num)
      num
    end

    def barrier_before_exit=(bool : Bool)
      MXNet.check_call LibMXNet.mx_kv_store_set_barrier_before_exit(@handle, bool)
    end

    def send_command_to_servers(head : Int32, body : String)
      MXNet.check_call LibMXNet.mx_kv_store_send_command_to_servers(@handle, head, body)
    end
  end
end
