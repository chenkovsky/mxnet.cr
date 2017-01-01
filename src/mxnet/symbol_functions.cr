module MXNet
  class Symbol::Function
    getter :name

    Functions = Hash(String, Symbol::Function).new
    init_functions.each do |f|
      Functions[f.name] = f
    end

    macro def_functions(*names)
        {% for name, index in names %}
        F_{{name.id}} = Functions["{{name.id}}"]
        {% end %}
    end

    def_functions :Activation, :BatchNorm, :BlockGrad, :element_mask
    def_functions :sum, :sum_axis, :broadcast_axis, :broadcast_to
    def_functions :Cast, :Concat, :Convolution, :Correlation, :Crop
    def_functions :_CrossDeviceCopy, :Custom, :Deconvolution, :Dropout
    def_functions :broadcast_plus, :broadcast_minus, :broadcast_mul, :broadcast_div
    def_functions :broadcast_power, :_Plus, :_Minus, :_Mul, :_Div, :_Power, :_Maximum
    def_functions :_Minimum, :_PlusScalar, :_MinusScalar, :_RMinusScalar, :_MulScalar
    def_functions :_DivScalar, :_RDivScalar, :_MaximumScalar, :_MinimumScalar, :_PowerScalar
    def_functions :_RPowerScalar, :ElementWiseSum, :abs, :sign, :round, :ceil, :floor, :square
    def_functions :sqrt, :rsqrt, :exp, :log, :cos, :sin, :Embedding, :FullyConnected
    def_functions :IdentityAttachKLSparseReg, :InstanceNorm, :L2Normalization, :LeakyReLU
    def_functions :softmax_cross_entropy, :LRN, :MakeLoss, :transpose, :expand_dims, :slice_axis
    def_functions :dot, :batch_dot, :_Native, :_NDArray, :Pad, :Pooling, :LinearRegressionOutput
    def_functions :MAERegressionOutput, :LogisticRegressionOutput, :Reshape, :Flatten, :RNN
    def_functions :ROIPooling, :uniform, :normal, :SequenceLast, :SequenceMask, :SequenceReverse
    def_functions :SliceChannel, :smooth_l1, :SoftmaxActivation, :SoftmaxOutput, :Softmax
    def_functions :SpatialTransformer, :SVMOutput, :SwapAxis, :UpSampling

    class Argument
      def initialize(@name : String, @type : String, @desc : String)
      end

      def to_s(io)
        io << "@param #{@name} : #{@type} ##{@desc}"
      end
    end

    def to_unsafe
      @handle
    end

    getter :name
    getter :key_var_num_args
    @key_var_num_args : String?

    def initialize(@handle : LibMXNet::AtomicSymbolCreator, @name : String, @desc : String, @ret_type : String, key_var_num_args : String?, @arguments : Array(Argument))
      @key_var_num_args = key_var_num_args.nil? || key_var_num_args.size == 0 ? nil : key_var_num_args
    end

    def to_s(io)
      io << "@func #{@name}\n"
      io << "@return #{@ret_type}\n"
      io << "@key_var_num_args #{@key_var_num_args}\n"
      io << "##{@desc}\n"
      @arguments.each do |arg|
        arg.to_s io
        io << "\n"
      end
    end

    def num_args
      @arguments.size
    end

    private def self.init_functions
      MXNet.check_call(LibMXNet.mx_symbol_list_atomic_symbol_creators(out list_size, out symbol_list))
      (0...list_size).map do |idx|
        handle = symbol_list[idx]
        MXNet.check_call(LibMXNet.mx_symbol_get_atomic_symbol_info(handle,
          out name,
          out desc,
          out num_args,
          out arg_names,
          out arg_types,
          out arg_descs,
          out key_var_num_args,
          out ret_type))
        name_s = String.new name
        desc_s = String.new desc
        args = (0...num_args).map do |idx|
          arg_name = String.new arg_names[idx]
          arg_type = String.new arg_types[idx]
          arg_desc = String.new arg_descs[idx]
          arg = Symbol::Function::Argument.new arg_name, arg_type, arg_desc
        end
        Symbol::Function.new handle, String.new(name), String.new(desc), String.new(ret_type), String.new(key_var_num_args), args
      end
    end
  end

  class Symbol
    enum ActType
      Relu
      Sigmoid
      Tanh
      Softrelu
    end

    def activation(act_type : ActType = ActType::Relu, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_Activation, name: name, act_type: act_type.to_s.downcase, data: self, attr: attr)
    end

    def batch_norm(eps : MXFloat = 0.001,
                   momentum : MXFloat = 0.9,
                   fix_gamma : Bool = true,
                   use_global_stats : Bool = false, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_BatchNorm, eps: eps,
        momentum: momentum,
        fix_gamma: fix_gamma,
        use_global_stats: use_global_stats,
        name: name, data: self, attr: attr)
    end

    def block_grad(name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_BlockGrad, data: self, name: name, attr: attr)
    end

    def sum(axis : Int32 | Array(Int32) | Nil = nil, keepdims : Bool = false, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_sum, src: self, name: name, axis: Shape.to_str(axis), keepdims: keepdims, attr: attr)
    end

    def broadcast_axis(axis : Int32 | Array(Int32) | Nil = nil, size : Int32 | Array(Int32) | Nil = nil, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_broadcast_axis, src: self, axis: Shape.to_str(axis), size: Shape.to_str(size), name: name, attr: attr)
    end

    def broadcast_to(shape : In32 | Array(Int32) | Nil = nil, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_broadcast_to, src: self, shape: Shape.to_str(shape), name: name, attr: attr)
    end

    def cast(dtype : MXType, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_Cast, data: self, dtype: dtype.to_s, name: name, attr: attr)
    end

    def self.concat(data : Array(Symbol), num_args : Int32, dim : Int32 = 1, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_concat, data: data, num_args: num_args, dim: dim, name: name, attr: attr)
    end

    def concat(other : Symbol, num_args : Int32, dim : Int32 = 1, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.concat([self, other], num_args: num_args, dim: dim, name: name, attr: attr)
    end

    enum CudnnTune
      Fastest
      Limited_Workspace
      Off
    end

    def convolution(kernel : Array(Int32),
                    num_filter : Int32,
                    weight : Symbol? = nil, bias : Symbol? = nil,
                    stride : Array(Int32) = [1, 1],
                    dilate : Array(Int32) = [1, 1],
                    pad : Array(Int32) = [0, 0],
                    num_group : Int32 = 1,
                    workspace : UInt64 = 1024,
                    no_bias : Bool = false,
                    cudnn_tune : CudnnTune = CudnnTune::Off,
                    cudnn_off : Bool = false,
                    name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_Convolution, data: self,
        weight: weight, bias: bias, kernel: Shape.to_str(kernel), num_filter: num_filter,
        stride: Shape.to_str(stride),
        dilate: Shape.to_str(dilate),
        pad: Shape.to_str(pad),
        num_group: num_group,
        workspace: workspace,
        no_bias: no_bias,
        cudnn_tune: cudnn_tune.to_s.downcase,
        cudnn_off: cudnn_off,
        name: name,
        attr: attr)
    end

    def correlation(data : Symbol,
                    kernel_size : Int32 = 1,
                    max_displacement : Int32 = 1,
                    stride1 : Int32 = 1,
                    stride2 : Int32 = 1,
                    pad_size : Int32 = 0,
                    is_multiply : Bool = true,
                    name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_Correlation, data1: self, data2: data,
        kernel_size: kernel_size,
        max_displacement: max_displacement,
        stride1: stride1,
        stride2: stride2,
        pad_size: pad_size,
        is_multiply: is_multiply,
        name: name,
        attr: attr)
    end

    def crop(offset : Array(Int32) = [0, 0],
             h_w : Array(Int32) = [0, 0],
             center_crop : Bool = false,
             crop_like : Symbol? = nil,
             name : String? = nil, attr : Hash(String, String)? = nil)
      # 按照文档应该还能传一个symbol。但是似乎不对
      if crop_like
        Symbol.create(Function::F_Crop, data: self, num_args: 2, crop_like: crop_like, offset: Shape.to_str(offset), center_crop: center_crop, name: name, attr: attr)
      else
        Symbol.create(Function::F_Crop, data: self, num_args: 1, offset: Shape.to_str(offset), h_w: Shape.to_str(h_w), center_crop: center_crop, name: name, attr: attr)
      end
    end

    private def cross_device_copy(name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F__CrossDeviceCopy, data: self, name: name, attr: attr)
    end

    def custom(op_type : String, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F__CrossDeviceCopy, data: self, op_type: op_type, name: name, attr: attr)
    end

    def deconvolution(weight : Symbol, bias : Symbol,
                      kernel : Array(Int32),
                      num_filter : Int32,
                      stride : Array(Int32) = [1, 1],
                      pad : Array(Int32) = [0, 0],
                      adj : Array(Int32) = [0, 0],
                      target_shape : Array(Int32) = [0, 0],
                      num_group : Int32 = 1,
                      workspace : UInt64 = 512,
                      no_bias : Bool = true,
                      name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_Deconvolution, data: self,
        weight: weight, bias: bias, kernel: Shape.to_str(kernel),
        num_filter: num_filter, stride: Shape.to_str(stride), pad: Shape.to_str(pad), adj: Shape.to_str(adj),
        target_shape: Shape.to_str(target_shape), num_group: num_group,
        workspace: workspace, no_bias: no_bias, name: name, attr: attr)
    end

    def dropout(p : Float32 = 0.5, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_Dropout, data: self, p: p, name: name, attr: attr)
    end

    def broadcast_plus(rhs : Symbol, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_broadcast_plus, lhs: self, rhs: rhs, name: name, attr: attr)
    end

    def broadcast_minus(rhs : Symbol, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_broadcast_minus, lhs: self, rhs: rhs, name: name, attr: attr)
    end

    def broadcast_mul(rhs : Symbol, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_broadcast_mul, lhs: self, rhs: rhs, name: name, attr: attr)
    end

    def broadcast_div(rhs : Symbol, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_broadcast_div, lhs: self, rhs: rhs, name: name, attr: attr)
    end

    def broadcast_power(rhs : Symbol, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_broadcast_power, lhs: self, rhs: rhs, name: name, attr: attr)
    end

    macro def_op(sym_func, inst_op)
        def {{inst_op.id}}(other : Symbol, name : String? = nil, attr : Hash(String, String)? = nil)
            Symbol.create(Function::F_{{sym_func.id}}, lhs: self, rhs: other, name: name, attr: attr)
        end
        def {{inst_op.id}}(other : Number, name : String? = nil, attr : Hash(String, String)? = nil)
            Symbol.create(Function::F_{{sym_func.id}}Scalar, self, scalar: other.to_s, name: name, attr: attr)
        end
    end

    macro def_class_op(sym_func, class_op)
        def self.{{class_op.id}}(lhs : Symbol, rhs : Symbol, name : String? = nil, attr : Hash(String, String)? = nil)
          Symbol.create(Function::F_{{sym_func.id}}, lhs: lhs, rhs: rhs, name: name, attr: attr)
        end

        def self.{{class_op.id}}(lhs : Symbol, rhs : Number, name : String? = nil, attr : Hash(String, String)? = nil)
            Symbol.create(Function::F_{{sym_func.id}}Scalar, src: lhs, scalar: rhs.to_s, name: name, attr: attr)
        end

        def self.{{class_op.id}}(lhs : Number, rhs : Symbol, name : String? = nil, attr : Hash(String, String)? = nil)
            Symbol.create(Function::F_{{sym_func.id}}Scalar, src: rhs, scalar: lhs.to_s, name: name, attr: attr)
        end
    end

    def_op :_Plus, :+
    def_op :_Minus, :-
    def_op :_Mul, :*
    def_op :_Div, :/
    def_op :_Power, :**
    def_class_op :_Maximum, :max
    def_class_op :_Minimum, :min

    def element_wise_sum(num_args : Int32, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_ElementWiseSum, num_args: num_args, name: name, attr: attr)
    end

    macro def_unary_op(*ops)
      {% for op, idx in ops %}
      def {{op.id}}(name : String? = nil, attr : Hash(String, String)? = nil)
        Symbol.create(Function::F_{{op.id}}, src: self, name: name, attr: attr)
      end
      {% end %}
    end

    def_unary_op :abs, :sign, :round
    def_unary_op :ceil, :floor, :square, :sqrt
    def_unary_op :rsqrt, :exp, :log, :cos, :sin

    def embedding(weight : Symbol, input_dim : Int32, output_dim : Int32, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_Embedding, data: self, weight: weight,
        input_dim: input_dim, output_dim: output_dim, name: name, attr: attr)
    end

    def fully_connected(num_hidden : Int32, weight : Symbol? = nil, bias : Symbol? = nil, no_bias : Bool = false, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_FullyConnected, data: self, weight: weight,
        bias: bias, num_hidden: num_hidden,
        no_bias: no_bias, name: name, attr: attr)
    end

    def self.fully_connected(num_hidden : Int32, weight : Symbol? = nil, bias : Symbol? = nil, no_bias : Bool = false, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_FullyConnected, weight: weight,
        bias: bias, num_hidden: num_hidden,
        no_bias: no_bias, name: name, attr: attr)
    end

    def identity_attach_kl_sparse_reg(sparseness_target : Float32 = 0.1, penalty : Float32 = 0.001, momentum : Float32 = 0.9, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_IdentityAttachKLSparseReg, data: self,
        sparseness_target: sparseness_target,
        penalty: penalty,
        momentum: momentum, name: name, attr: attr)
    end

    def instance_norm(gamma : Symbol, beta : Symbol, eps : Float32 = 0.001, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_InstanceNorm, data: self, gamma: gamma, beta: beta, eps: eps, name: name, attr: attr)
    end

    enum L2NormMode
      Channel
      Instance
      Spatial
    end

    def l2_normalization(eps : Float32 = 1e-10, mode : L2NormMode = L2NormMode::Instance, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_L2Normalization, data: self, eps: eps, mode: mode.to_s.downcase, name: name, attr: attr)
    end

    enum LeakyReLUActType
      Elu
      Leaky
      Prelu
      Rrelu
    end

    def leaky_relu(act_type : LeakyReLUActType = LeakyReLUActType::Leaky,
                   slope : Float32 = 0.25,
                   lower_bound : Float32 = 0.125,
                   upper_bound : Float32 = 0.334,
                   name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_LeakyReLU, data: self,
        act_type: act_type,
        slope: slope,
        lower_bound: lower_bound,
        upper_bound: upper_bound, name: name, attr: attr)
    end

    def self.softmax_cross_entropy(lhs, rhs, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_softmax_cross_entropy, lhs: lhs, rhs: rhs, name: name, attr: attr)
    end

    def softmax_cross_entropy(rhs, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_softmax_cross_entropy, lhs: self, rhs: rhs, name: name, attr: attr)
    end

    def lrn(nsize : Int32, alpha : Float32 = 0.0001, beta : Float32 = 0.75, knorm : Float32 = 2_f32, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_LRN, nsize: nsize, alpha: alpha, beta: beta, knorm: knorm, name: name, attr: attr)
    end

    enum Normalization
      Batch
      Null
      Valid
    end

    def make_loss(grad_scale : Float32 = 1_f32,
                  valid_thresh : Float32 = 0_f32,
                  normalization : Normalization = Normalization::Null, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_MakeLoss, data: self, grad_scale: grad_scale,
        valid_thresh: valid_thresh, normalization: normalization, name: name, attr: attr)
    end

    def transpose(axis : Array(Int32) = [] of Int32, name : String? = nil, attr : Hash(String, String)? = nil)
      # typo
      Symbol.create(Function::F_transpose, src: self, axis: Shape.to_str(axis), name: name, attr: attr)
    end

    def expand_dims(axis : Int32, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_expand_dism, src: self, axis: Shape.to_str(axis), name: name, attr: attr)
    end

    def slice_axis(axis : Int32, begin _begin : Int32, end _end : Int32, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_slice_axis, src: self, axis: axis, beign: _begin, end: _end, name: name, attr: attr)
    end

    def dot(rhs : Symbol, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_dot, lhs: self, rhs: rhs, name: name, attr: attr)
    end

    def batch_dot(rhs : Symbol, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_batch_dot, lhs: self, rhs: rhs, name: name, attr: attr)
    end

    # private def native
    # end
    # private def ndarray
    # end
    enum PadMode
      Constant
      Edge
    end

    def pad(mode : PadMode, pad_width : Array(Int32), constant_value : Float64 = 0.0, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_pad, data: self, mode: mode.to_s.downcase,
        pad_width: pad_width, constant_value: constant_value, name: name, attr: attr)
    end

    enum PoolType
      Avg
      Max
      Sum
    end
    enum PoolingConvention
      Full
      Valid
    end

    def pooling(kernel : Array(Int32),
                pool_type : PoolType,
                global_pool : Bool = false,
                pooling_convention : PoolingConvention = PoolingConvention::Valid,
                stride : Array(Int32) = [1, 1],
                pad : Array(Int32) = [0, 0], name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_Pooling, data: self,
        kernel: Shape.to_str(kernel),
        pool_type: pool_type.to_s.downcase,
        global_pool: global_pool,
        pooling_convention: pooling_convention.to_s.downcase,
        stride: Shape.to_str(stride),
        pad: Shape.to_str(pad), name: name, attr: attr)
    end

    def linear_regression_output(label : Symbol, grad_scale : Float32 = 1_f32, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_LinearRegressionOutput, data: self, label: label, grad_scale: grad_scale, name: name, attr: attr)
    end

    def mae_regression_output(label : Symbol, grad_scale : Floar32 = 1_f32, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_MAERegressionOutput, data: self, label: label, grad_scale: grad_scale, name: name, attr: attr)
    end

    def logistic_regression_output(label : Symbol, grad_scale : Float32 = 1_f32, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_LogisticRegressionOutput, data: self, label: label, grad_scale: grad_scale, name: name, attr: attr)
    end

    def reshape(target_shape : Array(Int32) = [0, 0],
                keep_highest : Bool = false,
                shape : Array(Int32) = [] of Int32,
                reverse : Bool = false, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_reshape, data: self,
        target_shape: Shape.to_str(target_shape),
        keep_highest: keep_highest,
        shape: Shape.to_str(shape),
        reverse: reverse, name: name, attr: attr
      )
    end

    def flatten(name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_Flatten, data: self, name: name, attr: attr)
    end

    enum RNNMode
      GRU
      LSTM
      RNN_Relu
      RNN_Tanh
    end

    def rnn(parameters : Symbol, state : Symbol, state_cell : Symbol,
            state_size : Int32, num_layers : Int32,
            mode : RNNMode,
            bidirectional : Bool = false, p : Float32 = 0_f32,
            state_outputs : Bool = false, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_RNN, data: self, parameters: parameters, state: state,
        state_cell: state_cell, stat_size: state_size,
        num_layers: num_layers, mode: mode.to_s.downcase,
        bidirectional: bidirectional, p: p, state_outputs: state_outputs, name: name, attr: attr)
    end

    def roi_pooling(rois : Symbol, pooled_size : Array(Int32), spatial_scale : Float32, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::ROIPooling, data: self, rois: rois,
        pooled_size: pooled_size, spatial_scale: spatial_scale, name: name, attr: attr)
    end

    def self.uniform(shape : Array(Int32), low : Float32 = 0_f32, high : Float32 = 1_f32, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_uniform, shape: Shape.to_str(shape), low: low, high: high, name: name, attr: attr)
    end

    def self.normal(shape : Array(Int32), loc : Float32 = 0_f32, scale : Float32 = 1_f32, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_normal, shape: Shape.to_str(shape), loc: loc, scale: scale, name: name, attr: attr)
    end

    def sequence_last(sequence_length : Symbol? = nil, name : String? = nil, attr : Hash(String, String)? = nil)
      if sequence_length
        Symbol.create(Functon::F_SequenceLast, data: self, sequence_length: sequence_length,
          use_sequence_length: true, name: name, attr: attr)
      else
        Symbol.create(Functon::F_SequenceLast, data: self, use_sequence_length: false, name: name, attr: attr)
      end
    end

    def sequence_mask(sequence_length : Symbol? = nil, value : Float32 = 0_f32, name : String? = nil, attr : Hash(String, String)? = nil)
      if sequence_length
        Symbol.create(Function::F_SequenceMask, data: self, sequence_length: sequence_length,
          use_sequence_length: true, value: value, name: name, attr: attr)
      else
        Symbol.create(Function::F_SequenceMask, data: self,
          use_sequence_length: false, value: value, name: name, attr: attr)
      end
    end

    def sequence_reverse(sequence_length : Symbol? = nil, name : String? = nil, attr : Hash(String, String)? = nil)
      if sequence_length
        Symbol.create(Functon::F_SequenceReverse, data: self, sequence_length: sequence_length,
          use_sequence_length: true, name: name, attr: attr)
      else
        Symbol.create(Functon::F_SequenceReverse, data: self, use_sequence_length: false, name: name, attr: attr)
      end
    end

    def slice_channel(num_outputs : Int32, axis : Int32 = 1, squeeze_axis : Bool = false, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_slice_channel, data: self, num_outputs: num_outputs,
        axis: axis,
        squeeze_axis: squeeze_axis, name: name, attr: attr)
    end

    def smooth_l1(name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_smooth_l1, src: self, name: name, attr: attr)
    end

    enum SoftmaxActivationMode
      Channel
      Instance
    end

    def softmax_activation(mode : SoftmaxActivationMode = SoftmaxActivationMode::Instance, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_SoftmaxActivation, data: self, mode: mode.to_s.downcase, name: name, attr: attr)
    end

    def softmax_output(label : Symbol? = nil, grad_scale : Float32 = 1_f32,
                       ignore_label : Float32 = -1_f32, multi_output : Bool = false,
                       use_ignore : Bool = false, preserve_shape : Bool = false,
                       normalization : Normalization = Normalization::Null, out_grad : Bool = false, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_SoftmaxOutput, data: self, label: label, grad_scale: grad_scale,
        ignore_label: ignore_label, multi_output: multi_output,
        use_ignore: use_ignore, preserve_shape: preserve_shape,
        normalization: normalization.to_s.downcase, out_grad: out_grad, name: name, attr: attr)
    end

    enum SpatialTransformerType
      Affine
    end
    enum SpatialTransformerSampleType
      Bilinear
    end

    def spatial_transformer(loc : Symbol, target_shape : Array(Int32),
                            transform_type : SpatialTransformerType = SpatialTransformerType::Affine,
                            sampler_type : SpatialTransformerSampleType = SpatialTransformerSampleType::Bilinear, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(data: self, loc: loc, target_shape: Shape.to_str(target_shape),
        transform_type: transform_type.to_s.downcase,
        sampler_type: sampler_type.to_s.downcase, name: name, attr: attr)
    end

    def svm_output(label : Symbol, margin : Float32 = 1_f32,
                   regularization_coefficient : Float32 = 1_f32,
                   use_linear : Bool = false, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(data: self, label: label, margin: margin,
        regularization_coefficient: regularization_coefficient,
        use_linear: use_linear, name: name, attr: attr)
    end

    def swap_axis(dim1 : Int32 = 0, dim2 : Int32 = 0, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(data: self, dim1: dim1, dim2: dim2, name: name, attr: attr)
    end

    enum UpSamplingType
      Bilinear
      Nearest
    end

    enum UpSamplingMultiInputMode
      Concat
      Sum
    end

    def up_sampling(scale : Int32, sampler_type : UpSamplingType,
                    num_args : Int32,
                    num_filter : Int32 = 0,
                    multi_input_mode : UpSamplingMultiInputMode = UpSamplingMultiInputMode::Concat,
                    workspace : UInt64 = 512, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_UpSampling, data: self, scale: scale,
        sampler_type: sampler_type.to_s.downcase,
        num_args: num_args,
        multi_input_mode: multi_input_mode.to_s.downcase,
        workspace: workspace, name: name, attr: attr)
    end

    def self.up_sampling(data : Array(Symbol), scale : Int32, sampler_type : UpSamplingType,
                         num_args : Int32,
                         num_filter : Int32 = 0,
                         multi_input_mode : UpSamplingMultiInputMode = UpSamplingMultiInputMode::Concat,
                         workspace : UInt64 = 512, name : String? = nil, attr : Hash(String, String)? = nil)
      Symbol.create(Function::F_UpSampling, self, scale: scale,
        sampler_type: sampler_type.to_s.downcase,
        num_args: num_args,
        multi_input_mode: multi_input_mode.to_s.downcase,
        workspace: workspace, name: name, attr: attr)
    end
  end
end

abstract struct Number
  macro def_op(sym_func, inst_op)
    def {{sym_func.id}}(rhs : MXNet::Symbol, name : String? = nil, attr : Hash(String, String)? = nil)
      MXNet::Symbol.create(Function::F_{{sym_func.id}}, scalar: self, src: rhs, name: name, attr: attr)
    end
  end

  def_op :_RPlusScalar, :+
  def_op :_RMinusScalar, :-
  def_op :_RMulScalar, :*
  def_op :_RDivScalar, :/
  def_op :_RPowerScalar, :**
end
