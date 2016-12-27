module MXNet
  abstract class Optimizer
    OPTIMIZER_MAPPING["CCSGD"] = CCSGD

    class CCSGD < Optimizer
      @learning_rate : Float32
      @momentum : Float32
      @wd : Float32
      @rescale_gradient : Float32
      @clip_gradient : Float32
      @lr_scheduler : LRScheduler?

      CCSGD_JSON = Optimizer_JSON + {
        learning_rate:    Float32,
        momentum:         Float32,
        wd:               Float32,
        rescale_gradient: Float32,
        clip_gradient:    Float32,
        lr_scheduler:     {nilable: true, type: LRScheduler},
      }
      JSON.mapping(CCSGD_JSON)

      def initialize(@learning_rate = 0.01_f32,
                     @momentum = 0.0_f32,
                     @wd = 0.0001_f32,
                     @rescale_gradient = 1.0_f32,
                     @clip_gradient = -1_f32,
                     @lr_scheduler = nil)
        if !@lr_scheduler.nil?
          @lr_scheduler.base_lr = @learning_rate
        end
        @opt_creator = OptimizerCreator.null
        @opt_handle = OptimizerHandle.null
        check_call LibMXNet.mx_optimizer_find_creator("ccsgd", out @opt_creator)
        param_keys = ["momentum", "rescale_grad", "clip_gradient"].map &.to_unsafe
        param_vals = [@momentum, @rescale_gradient, @clip_gradient]
        check_call LibMXNet.mx_optimizer_create_optimizer(@opt_creator, param_keys.size, param_keys, param_vals, out @opt_handle)
      end

      def update(index : Int32, weight : NDArray, grad : NDArray, state)
        lr = if @lr_scheduler.nil?
               @learning_rate
             else
               scheduler_lr = @lr_scheduler[@num_update]
               update_count index
               scheduler_lr
             end
        lr *= @lr_scale.fetch(index, 1_f32)
        wd = get_wd(index, @wd)
        check_call LibMXNet.mx_optimizer_update(@opt_handle, index, weight, grad, lr, wd)
      end

      def create_state(index : Int32, weight : NDArray)
        nil
      end
    end
  end
end
