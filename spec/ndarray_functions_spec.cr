require "./spec_helper"

describe MXNet do
  describe "ndarray functions" do
    it "sqrt" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      e1, e2, e3 = arr.sqrt.to_a.as(Array(Float32))
      e1.should be_close(1, 0.00001)
      e2.should be_close(1.41421, 0.00001)
      e3.should be_close(1.73205, 0.00001)
    end

    it "rsqrt" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      e1, e2, e3 = arr.rsqrt.to_a.as(Array(Float32))
      e1.should be_close(1, 0.00001)
      e2.should be_close(1/1.41421, 0.00001)
      e3.should be_close(1/1.73205, 0.00001)
    end

    it "norm" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      norm_arr = arr.norm.to_a.as(Array(Float32))
      norm_arr[0].should be_close(Math.sqrt((1 + 4 + 9)), 0.00001)
    end

    it "abs" do
      arr = MXNet::NDArray.array([-1, -2, -3], shape: [3, 1])
      e1, e2, e3 = arr.abs.to_a.as(Array(Float32))
      e1.should be_close(1, 0.00001)
      e2.should be_close(2, 0.00001)
      e3.should be_close(3, 0.00001)
    end

    it "sign" do
      arr = MXNet::NDArray.array([1, -2, -3], shape: [3, 1])
      e1, e2, e3 = arr.sign.to_a.as(Array(Float32))
      e1.should be_close(1, 0.00001)
      e2.should be_close(-1, 0.00001)
      e3.should be_close(-1, 0.00001)
    end

    it "round" do
      arr = MXNet::NDArray.array([1.5, -2.4, 1.4, -2.5], shape: [4, 1])
      e1, e2, e3, e4 = arr.round.to_a.as(Array(Float32))
      e1.should be_close(2, 0.00001)
      e2.should be_close(-2, 0.00001)
      e3.should be_close(1, 0.00001)
      e4.should be_close(-3, 0.00001)
    end

    it "ceil" do
      arr = MXNet::NDArray.array([1.5, -2.4, 1.4, -2.5], shape: [4, 1])
      e1, e2, e3, e4 = arr.ceil.to_a.as(Array(Float32))
      e1.should be_close(2, 0.00001)
      e2.should be_close(-2, 0.00001)
      e3.should be_close(2, 0.00001)
      e4.should be_close(-2, 0.00001)
    end

    it "floor" do
      arr = MXNet::NDArray.array([1.5, -2.4, 1.4, -2.5], shape: [4, 1])
      e1, e2, e3, e4 = arr.floor.to_a.as(Array(Float32))
      e1.should be_close(1, 0.00001)
      e2.should be_close(-3, 0.00001)
      e3.should be_close(1, 0.00001)
      e4.should be_close(-3, 0.00001)
    end

    it "square" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      e1, e2, e3 = arr.square.to_a.as(Array(Float32))
      e1.should be_close(1, 0.00001)
      e2.should be_close(4, 0.00001)
      e3.should be_close(9, 0.00001)
    end

    it "exp" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      e1, e2, e3 = arr.exp.to_a.as(Array(Float32))
      e1.should be_close(Math.exp(1), 0.00001)
      e2.should be_close(Math.exp(2), 0.00001)
      e3.should be_close(Math.exp(3), 0.00001)
    end

    it "log" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      e1, e2, e3 = arr.log.to_a.as(Array(Float32))
      e1.should be_close(Math.log(1), 0.00001)
      e2.should be_close(Math.log(2), 0.00001)
      e3.should be_close(Math.log(3), 0.00001)
    end

    it "cos" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      e1, e2, e3 = arr.cos.to_a.as(Array(Float32))
      e1.should be_close(Math.cos(1), 0.00001)
      e2.should be_close(Math.cos(2), 0.00001)
      e3.should be_close(Math.cos(3), 0.00001)
    end

    it "sin" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      e1, e2, e3 = arr.sin.to_a.as(Array(Float32))
      e1.should be_close(Math.sin(1), 0.00001)
      e2.should be_close(Math.sin(2), 0.00001)
      e3.should be_close(Math.sin(3), 0.00001)
    end

    it "max" do
      arr = MXNet::NDArray.array([1, 2, 3, -1], shape: [4, 1])
      max_arr = arr.max.to_a.as(Array(Float32))
      max_arr[0].should be_close(3, 0.00001)
    end

    it "min" do
      arr = MXNet::NDArray.array([1, 2, 3, -1], shape: [4, 1])
      arr2 = arr.min.to_a.as(Array(Float32))
      arr2[0].should be_close(-1, 0.00001)
    end

    it "sum" do
      arr = MXNet::NDArray.array([1, 2, 3, -1], shape: [4, 1])
      arr2 = arr.sum.to_a.as(Array(Float32))
      arr2[0].should be_close(5, 0.00001)
    end

    it "argmax_channel" do
      arr = MXNet::NDArray.array([1, 2, 4, 3, 5, 6], shape: [3, 2])
      arr2 = arr.argmax_channel.to_a.as(Array(Float32))
      arr2.should eq([1, 0, 1])
    end

    it "plus" do
      arr = MXNet::NDArray.array([1, 2, 4], shape: [3, 1])
      arr2 = MXNet::NDArray.array([3, 5, 6], shape: [3, 1])
      arr3 = arr + arr2
      arr3.to_a.as(Array(Float32)).should eq([4, 7, 10])
    end

    it "plus Float32" do
      arr = MXNet::NDArray.array([1, 2, 4], shape: [3, 1])
      arr2 = arr + 1_f32
      arr2.to_a.as(Array(Float32)).should eq([2, 3, 5])
    end

    it "minus" do
      arr = MXNet::NDArray.array([1, 2, 4], shape: [3, 1])
      arr2 = MXNet::NDArray.array([3, 5, 6], shape: [3, 1])
      arr3 = arr - arr2
      arr3.to_a.as(Array(Float32)).should eq([-2, -3, -2])
    end

    it "mul" do
      arr = MXNet::NDArray.array([1, 2, 4], shape: [3, 1])
      arr2 = MXNet::NDArray.array([3, 5, 6], shape: [3, 1])
      arr3 = arr * arr2
      arr3.to_a.as(Array(Float32)).should eq([3, 10, 24])
    end

    it "div" do
      arr = MXNet::NDArray.array([1, 2, 4], shape: [3, 1])
      arr2 = MXNet::NDArray.array([3, 5, 6], shape: [3, 1])
      arr3 = arr / arr2
      e1, e2, e3 = arr3.to_a.as(Array(Float32))
      e1.should be_close(1/3.0, 0.00001)
      e2.should be_close(2/5.0, 0.00001)
      e3.should be_close(4/6.0, 0.00001)
    end

    it "dot" do
      arr = MXNet::NDArray.array([1, 2, 3, 4], shape: [2, 2])
      arr2 = MXNet::NDArray.array([5, 6, 7, 8], shape: [2, 2])
      arr3 = arr.dot(arr2).to_a.as(Array(Float32))
      arr3.should eq([19, 22, 43, 50])
    end

    it "choose_element_0index" do
      arr = MXNet::NDArray.array([1, 2, 4, 3, 5, 6], shape: [3, 2])
      arr2 = MXNet::NDArray.array([1, 0, 1], shape: [3])
      arr3 = arr.choose_element_0index(arr2).to_a.as(Array(Float32))
      arr3.should eq([2, 4, 6])
    end

    it "onehotEncode" do
      arr = MXNet::NDArray.array([0, 0, 3, 1], shape: [4])
      arr2 = MXNet::NDArray.empty([4, 4])
      arr.onehot_encode arr2
      arr3 = arr2.to_a.as(Array(Float32))
      arr3.should eq([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0])
    end

    it "clip" do
      arr = MXNet::NDArray.array([1, 2, 4, 3, 5, 6], shape: [3, 2])
      arr2 = arr.clip(2_f32, 4_f32).to_a.as(Array(Float32))
      arr2.should eq([2, 2, 4, 3, 4, 4])
    end

    it "random_uniform" do
      arr = MXNet::NDArray.empty([4, 3])
      arr.random_uniform(0.2_f32, 0.8_f32)
    end

    it "random_gaussian" do
      arr = MXNet::NDArray.empty([4, 3])
      arr.random_gaussian(0.2_f32, 0.8_f32)
    end

    it "plus!" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      arr2 = MXNet::NDArray.array([3, 4, 5], shape: [3, 1])
      arr2.plus! arr
      arr2.to_a.as(Array(Float32)).should eq([4, 6, 8])
    end

    it "- Float" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      arr2 = arr - 1_f32
      arr2.to_a.as(Array(Float32)).should eq([0, 1, 2])
    end

    it "minus! NDArray" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      arr2 = MXNet::NDArray.array([3, 4, 5], shape: [3, 1])
      arr2.minus! arr
      arr2.to_a.as(Array(Float32)).should eq([2, 2, 2])
    end

    it "minus! Float" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      arr.minus! 1_f32
      arr.to_a.as(Array(Float32)).should eq([0, 1, 2])
    end

    it "* MXFloat" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      arr2 = arr * 2_f32
      arr2.to_a.as(Array(Float32)).should eq([2, 4, 6])
    end

    it "mul! NDArray" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      arr2 = MXNet::NDArray.array([2, 3, 4], shape: [3, 1])
      arr.mul! arr2
      arr.to_a.as(Array(Float32)).should eq([2, 6, 12])
    end

    it "mul! MXFloat" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      arr.mul! 2_f32
      arr.to_a.as(Array(Float32)).should eq([2, 4, 6])
    end

    it "neg NDArray" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      arr = -arr
      arr.to_a.as(Array(Float32)).should eq([-1, -2, -3])
    end

    it "/ NDArray" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      arr2 = MXNet::NDArray.array([2, 4, 6], shape: [3, 1])
      arr3 = arr2 / arr
      arr3.to_a.as(Array(Float32)).should eq([2, 2, 2])
    end

    it "/ MXFloat" do
      arr = MXNet::NDArray.array([2, 4, 6], shape: [3, 1])
      arr2 = arr / 2_f32
      arr2.to_a.as(Array(Float32)).should eq([1, 2, 3])
    end

    it "div! NDArray" do
      arr = MXNet::NDArray.array([1, 2, 3], shape: [3, 1])
      arr2 = MXNet::NDArray.array([2, 4, 6], shape: [3, 1])
      arr2.div! arr
      arr2.to_a.as(Array(Float32)).should eq([2, 2, 2])
    end

    it "div! MXFloat" do
      arr = MXNet::NDArray.array([2, 4, 6], shape: [3, 1])
      arr.div! 2_f32
      arr.to_a.as(Array(Float32)).should eq([1, 2, 3])
    end

    it "Float + NDArray" do
      arr = MXNet::NDArray.array([2, 4, 6], shape: [3, 1])
      arr2 = 1_f32 + arr
      arr2.to_a.as(Array(Float32)).should eq([3, 5, 7])
    end

    it "Float - NDArray" do
      arr = MXNet::NDArray.array([2, 4, 6], shape: [3, 1])
      arr2 = 1_f32 - arr
      arr2.to_a.as(Array(Float32)).should eq([-1, -3, -5])
    end

    it "Float * NDArray" do
      arr = MXNet::NDArray.array([2, 4, 6], shape: [3, 1])
      arr2 = 2_f32 * arr
      arr2.to_a.as(Array(Float32)).should eq([4, 8, 12])
    end

    it "Float / NDArray" do
      arr = MXNet::NDArray.array([2, 4, 6], shape: [3, 1])
      arr2 = 12_f32 / arr
      arr2.to_a.as(Array(Float32)).should eq([6, 3, 2])
    end
  end
end
