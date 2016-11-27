module MXNet
  module Metric
    abstract class EvalMetric
      # Base class of all evaluation metrics
      @name : String
      @num_inst : Int32
      @sum_metric : Float32

      def initialize(@name)
        @num_inst = 0
        @sum_metric = 0.0_f32
      end

      def reset
        @num_inst = 0
        @sum_metric = 0.0_f32
      end

      abstract def update(label : Array(NDArray), pred : Array(NDArray))
    end

    class Accuracy < EvalMetric
      def initialize
        super("accuracy")
      end

      def update(labels : Array(NDArray), preds : Array(NDArray))
        check_labels_size(labels, preds)
        preds.zip(labels) do |pred, label|
          pred_label = NDArray.argmax_channel(pred)
          check_label_shape(label, pred_label)
          label.to_a.zip(pred_label.to_a) do |label_elem, pred_elem|
            @sum_metric += 1 if label_elem == pred_elem
          end
          @num_inst += pred_label.shape[0]
        end
      end
    end

    class TopKAccuracy < EvalMetric
      def initialize(@top_k : Int32)
        super("top_k_accuracy")
      end

      def update(labels : Array(NDArray), preds : Array(NDArray))
        check_labels_size(labels, preds)
        preds.zip(labels) do |pred, label|
          pred_shape = pred.shape
          dims = pred_shape.size
          raise MXError.new "Predictions should be no more than 2 dims." if dims > 2
          label_arr = label.to_a
          num_samples = pred_shape[0]
          if dims == 1
            pred_arr = arg_sort(pred.to_a) { |e| -e }
            check_labels_size label_arr, pred_arr
            label_arr.zip(pred_arr) do |l, p|
              @sum_metric += 1 if l == p
            end
          else
            num_classes = pred_shape[1]
            pred_arr = pred.to_a.in_groups_of(num_classes, 0.0_f32).map { |a|
              arg_sort(a) { |e| -e }
            }
            check_labels_size label_arr, pred_arr
            top_k = {@top_k, num_classes}.max
            (0...top_k).each do |j|
              label_arr.zip(pred_arr.map { |arr| arr[j] }) do |l, p|
                @sum_metric += 1 if l == p
              end
            end
          end
          @num_inst += num_samples
        end
      end
    end

    class F1 < EvalMetric
      def initialize
        super("f1")
      end

      def update(labels : Array(NDArray), preds : Array(NDArray))
        check_labels_size(labels, preds)
        preds.zip(labels) do |pred, label|
          pred_label = NDArray.argmax_channel pred
          check_label_shape(label, pred_label)
          label_arr = label.to_a
          unique = Set.new label_arr
          raise MXError.new "F1 currently only support binary classification." if unique > 2
          true_positives, false_positives, false_negatives = 0.0, 0.0, 0.0
          label_arr.zip(pred_label.to_a) do |label_elem, pred_elem|
            true_positives += 1 if pred_elem == 1 && label_elem == 1
            false_positives += 1 if pred_elem == 1 && label_elem == 0
            false_positives += 1 if pred_elem == 0 && label_elem == 1
          end
          precision = if true_positives + false_positives > 0
                        true_positives / (true_positives + false_positives)
                      else
                        0.0
                      end
          recall = if true_positives + false_negatives > 0
                     true_positives / (true_positives + false_negatives)
                   else
                     0.0
                   end
          f1_score = if precision + recall > 0
                       (2*precision*recall)/(precision + recall)
                     else
                       0.0
                     end
          @sum_metric += f1_score
          @num_inst += 1
        end
      end
    end

    class MAE < EvalMetric
      def initialize
        super("mae")
      end

      def update(labels : Array(NDArray), preds : Array(NDArray))
        check_labels_size(labels, preds)
        labels.zip(preds) do |label, pred|
          label_arr = label.to_a
          pred_arr = pred.to_a
          raise MXError.new "label_arr.size should be eq to pred_arr.size" unless label_arr.size == pred_arr.size
          sum = 0
          label_arr.zip(pred_arr) do |l, p|
            sum += Math.abs(l - p)
          end
          @sum_metric += sum / label_arr.size
          @num_inst += 1
        end
      end
    end

    class MSE < EvalMetric
      def initialize
        super("mse")
      end

      def update(labels : Array(NDArray), preds : Array(NDArray))
        check_labels_size(labels, preds)
        labels.zip(preds) do |label, pred|
          label_arr = label.to_a
          pred_arr = pred.to_a
          raise MXError.new "label_arr.size should be eq to pred_arr.size" unless label_arr.size == pred_arr.size
          sum = 0
          label_arr.zip(pred_arr) do |l, p|
            sum += (l - p)*(l - p)
          end
          @sum_metric += sum / label_arr.size
          @num_inst += 1
        end
      end
    end

    class RMSE < EvalMetric
      def initialize
        super("rmse")
      end

      def update(labels : Array(NDArray), preds : Array(NDArray))
        check_labels_size(labels, preds)
        labels.zip(preds) do |label, pred|
          label_arr = label.to_a
          pred_arr = pred.to_a
          raise MXError.new "label_arr.size should be eq to pred_arr.size" unless label_arr.size == pred_arr.size
          sum = 0
          label_arr.zip(pred_arr) do |l, p|
            sum += (l - p)*(l - p)
          end
          @sum_metric += Math.sqrt(sum / label_arr.size)
          @num_inst += 1
        end
      end
    end

    class CrossEntropy < EvalMetric
      def initialize
        super("cross-entropy")
      end

      def update(labels : Array(NDArray), preds : Array(NDArray))
        check_labels_size(labels, preds)
        labels.zip(preds) do |label, pred|
          label_arr = label.to_a
          pred_arr = pred.to_a.in_groups_of(label_arr.shape[0])
          raise MXError.new "label_arr.size should be eq to pred_arr.size" unless label_arr.size == pred_arr.size
          probs = label_arr.enum_with_index.map { |e, i| pred_arr[i][e] }
          @sum_metric += probs.map { |p| Math.log(p) }.sum
          @num_inst += labels.shape[0]
        end
      end
    end

    class CustomMetric < EvalMetric
      def initialize(name, @feval : (NDArray, NDArray -> Float32))
        super(name)
      end

      def update(labels : Array(NDArray), preds : Array(NDArray))
        check_labels_size(labels, preds)
        labels.zip(preds).each do |label, pred|
          @sum_metric += @feval.call(label, pred)
          @num_inst += 1
        end
      end
    end

    protected def self.check_labels_size(labels : Array(_), preds : Array(_))
      raise MXError.new("size of labels #{labels.size} does not match size of predictions #{preds.size}") unless labels.size == preds.size
    end
    protected def self.check_label_shape(label : NDArray, pred : NDArray)
      raise MXError.new "label #{label.shape} and pred #{pred.shape} should have the same length" unless label.shape == preds.shape
    end
  end

  def self.arg_sort(arr : Array(_)) : Array(Int32)
    arr.enum_with_index.map { |e_i| e_i }.sort_by! { |e_i| yield e_i[0] }.map { |e_i| e_i[1] }
  end
end
