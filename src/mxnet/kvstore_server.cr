module MXNet
  class KVStore
    class Server
      # a server node for the key value store
      @kv_store : KVStore
      @@logger = Logger.new STDOUT
      alias MXKVStoreServerController = Int32, String -> Void
      @controller : MXKVStoreServerController

      def initialize(@kv_store)
        @controller = ->(cmd_id : Int32, cmd_body : String) {
          @@logger.debug("Receive cmd_id: #{cmd_id}, cmd_body: #{cmd_body}")
          if cmd_id == 0
            optimizer = Serializer.serializer.deserialize(Serializer.decode_base64_string(cmd_body))
            @kv_store.optimizer = optimizer
          else
            @@logger.warn "Server #{@kv_store.rank}, unknown command (#{cmd_id}, #{cmd_body})"
          end
        }
      end

      def run
        boxed_data = Box.box(@controller)
        MXNet.check_call LibMXNet.mx_kv_store_run_server(@kv_store.handle, ->(cmd_id, cmd_body, data) {
          data_as_callback = Box(typeof(callback)).unbox(data)
          data_as_callback.call(cmd_id, String.new cmd_body)
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
