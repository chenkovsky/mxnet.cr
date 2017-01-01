require "./spec_helper"

describe MXNet do
  describe "ndarray" do
    it "zeros" do
      arr = MXNet::NDArray.zeros([2, 3], MXNet::Context.cpu)
      [0, 0, 0, 0, 0, 0].should eq(arr.to_a)
      arr.size.should eq(6)
    end
    it "ones" do
      arr = MXNet::NDArray.ones([2, 3], MXNet::Context.cpu)
      [1, 1, 1, 1, 1, 1].should eq(arr.to_a)
    end
    it "fill scalar" do
      arr = MXNet::NDArray.ones([2, 3], MXNet::Context.cpu)
      arr.fill 2
      [2, 2, 2, 2, 2, 2].should eq(arr.to_a)
    end
    it "fill array" do
      arr = MXNet::NDArray.ones([2, 3], MXNet::Context.cpu)
      tmp = [1, 2, 3, 4, 5, 6]
      arr.fill tmp
      tmp.should eq(arr.to_a)
    end
    it "fill ndarray" do
      arr = MXNet::NDArray.ones([2, 3], MXNet::Context.cpu)
      tmp = [1, 2, 3, 4, 5, 6]
      arr.fill tmp
      arr2 = MXNet::NDArray.ones([2, 3], MXNet::Context.cpu)
      arr2.fill arr
      tmp.should eq(arr2.to_a)
    end
    it "to_scalar" do
      arr = MXNet::NDArray.ones([1], MXNet::Context.cpu)
      1.should eq(arr.to_scalar)
    end

    it "array" do
      tmp = [1, 2, 3, 4, 5, 6]
      arr = MXNet::NDArray.array(tmp, [2, 3])
      arr.shape.to_a.should eq([2, 3])
      tmp.should eq(arr.to_a)
    end

    it "concat" do
      tmp = [1, 2, 3, 4, 5, 6]
      arr = MXNet::NDArray.array(tmp, [2, 3])
      arr2 = MXNet::NDArray.array(tmp, [2, 3])
      arr3 = MXNet::NDArray.concat(arr, arr2)
      arr3.to_a.should eq(tmp + tmp)
    end

    it "save & load" do
      tmp = [1, 2, 3, 4, 5, 6]
      arr = MXNet::NDArray.array(tmp, [2, 3])
      MXNet::NDArray.save(".spec_save_load", [arr])
      data = MXNet::NDArray.load(".spec_save_load")
      data.is_a?(Array(MXNet::NDArray)).should eq(true)
      data2 = data.as(Array(MXNet::NDArray))
      data2.size.should eq(1)
      data2[0].to_a.should eq(tmp)

      MXNet::NDArray.save(".spec_save_load", {"my_arr" => arr})
      data = MXNet::NDArray.load(".spec_save_load")
      data.is_a?(Hash(String, MXNet::NDArray)).should eq(true)
      data2 = data.as(Hash(String, MXNet::NDArray))
      data2.size.should eq(1)
      data2["my_arr"].to_a.should eq(tmp)
      FileUtils.rm ".spec_save_load"
    end

    it "serialize & deserialize" do
      tmp = [1, 2, 3, 4, 5, 6]
      arr = MXNet::NDArray.array(tmp, [2, 3])
      bytes = arr.serialize
      arr2 = MXNet::NDArray.deserialize bytes
      arr.to_a.should eq(arr2.to_a)
    end

    it "slice" do
      tmp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
      arr = MXNet::NDArray.array(tmp, [6, 2])
      arr[(0...2)].to_a.should eq([1, 2, 3, 4])
      arr[1, 2].to_a.should eq([3, 4, 5, 6])
      arr[3].to_a.should eq([7, 8])
    end

    it "slice fill" do
      tmp = [1, 2, 3, 4]
      arr = MXNet::NDArray.array(tmp, [4, 1])
      arr[1..3] = 5
      arr.to_a.should eq([1, 5, 5, 5])
      arr[1, 2] = 4
      arr.to_a.should eq([1, 4, 4, 5])
      arr[1..3] = MXNet::NDArray.array([6, 7, 8], [3, 1])
      arr.to_a.should eq([1, 6, 7, 8])
      arr[1..3] = [9, 10, 11]
      arr.to_a.should eq([1, 9, 10, 11])
    end

    it "reshape" do
      tmp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
      arr = MXNet::NDArray.array(tmp, [6, 2])
      arr2 = arr.reshape([3, 4])
      arr2.shape.to_a.should eq([3, 4])
    end

    it "context" do
      tmp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
      arr = MXNet::NDArray.array(tmp, [6, 2])
      arr.context.should eq(MXNet::Context.default_ctx)
    end
    it "===" do
      tmp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
      arr = MXNet::NDArray.array(tmp, [6, 2])
      arr2 = MXNet::NDArray.array(tmp, [6, 2])
      (arr == arr2).should eq(true)
    end

    it "dtype" do
      tmp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
      arr = MXNet::NDArray.array(tmp, [6, 2])
      arr.dtype.should eq(MXNet::MXType::Float32_T)
    end
  end
end
