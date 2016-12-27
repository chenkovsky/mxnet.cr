module MXNet
  class KVStore
    class Server
      # a server node for the key value store
      @kv_store : KVStore
      @@logger = Logger.new STDOUT
      alias MXKVStoreServerController = Int32, UInt8* -> Nil
      @controller : MXKVStoreServerController

      def initialize(@kv_store)
        @controller = ->(cmd_id : Int32, cmd_body : UInt8*) {
          @@logger.debug "Receive cmd_id: #{cmd_id}"
          if cmd_id == 0
            optimizer = Optimizer.deserialize cmd_body
            @kv_store.optimizer = optimizer
            nil
          else
            @@logger.warn "Server #{@kv_store.rank}, unknown command (#{cmd_id})"
            nil
          end
        }
      end

      def run
        boxed_data = Box.box(@controller)
        MXNet.check_call LibMXNet.mx_kv_store_run_server(@kv_store, ->(cmd_id, cmd_body, data) {
          data_as_callback = Box(MXKVStoreServerController).unbox(data)
          data_as_callback.call(cmd_id, cmd_body)
        }, boxed_data)
      end

      def self.start
        # TODO support dieIfOthersGoOutTimeout
        MXNet.check_call LibMXNet.mx_kv_store_is_worker_node(out is_worker)
        # raise MXError.new "cannot start kv-store server on worker node" unless is_worker == 0
        if is_worker == 0
          kv_store = KVStore.create(KVStore::Type::Dist)
          server = Server.new kv_store
          server.run
          exit
        end
      end

      start
    end
  end
end
