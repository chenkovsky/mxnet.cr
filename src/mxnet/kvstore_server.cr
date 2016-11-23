module MXNet
  class KVStoreServer
    @kv_store : KVStore
    @@logger = Logger.new STDOUT
    alias MXKVStoreServerController = Int32, String -> Void
    @controller : MXKVStoreServerController
    @mx_kv_store_controller : LibMXNet::MXKVStoreServerController

    private def mx_kv_store_run_server(handle : KVStoreHandle, &controller : MXKVStoreServerController)
      @mx_kv_store_controller = controller
      boxed_data = Box.box(controller)
      check_call LibMXNet.mx_kv_store_run_server(handle, ->(cmd_id, cmd_body, data) {
        data_as_callback = Box(typeof(callback)).unbox(data)
        data_as_callback.call(cmd_id, String.new cmd_body)
      }, boxed_data)
    end

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
      mx_kv_store_run_server(@kv_store.handle, @controller)
    end

    def self.start
      # TODO support dieIfOthersGoOutTimeout
      is_worker = 0
      check_call LibMXNet.mx_kv_store_is_worker_node(out is_worker)
      # raise MXError.new "cannot start kv-store server on worker node" unless is_worker == 0
      if is_worker == 0
        kv_store = KVStore.create(KVStoreType::Dist)
        server = KVStoreServer.new kv_store
        server.run
        exit
      end
    end

    start
  end
end
