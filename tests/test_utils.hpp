#ifndef _TEST_UTILS_H_
#define _TEST_UTILS_H_

namespace fourierconvolution {

  template <int value = 42>
  struct set_to_float {

    float operator()(){

      return float(value);
    
    }
  
  };

  template<typename in_type, typename out_type = in_type>
  struct diff_squared {

    out_type operator()(const in_type& _first, const in_type& _second){

      out_type value = _first - _second;
      return (value*value);
    
    }
  
  };



  struct ramp
  {
    size_t value;

    ramp():
      value(0){};
  
    float operator()(){

      return value++;
    
    }
  
  };
  

  template <typename stack_type, typename value_policy = ramp>
  struct stack_fixture {

  
    stack_type stack;
    stack_type kernel;

    template <typename T>
    stack_fixture(const std::vector<T>& _stack_shape,
		  const std::vector<T>& _kernel_shape):
      stack(_stack_shape),
      kernel(_kernel_shape){

      value_policy operation;
      std::fill(kernel.data(),kernel.data()+kernel.num_elements(),0);
      std::generate(stack.data(),stack.data()+stack.num_elements(),operation);
    
    }
  
  };


  
};

template <typename stack_type>
double l2norm(const stack_type& _reference, const stack_type& _data){
  double l2norm = std::inner_product(_data.data(),
				     _data.data() + _data.num_elements(),
				     _reference.data(),
				     0.,
				     std::plus<double>(),
				     fc::diff_squared<float,double>()
				     );

  double value = std::sqrt(l2norm)/_data.num_elements();

  return value;
}

template <typename stack_type>
double l2norm_by_nvidia(const stack_type& _reference, const stack_type& _data){
  double l2norm = std::inner_product(_data.data(),
				     _data.data() + _data.num_elements(),
				     _reference.data(),
				     0.,
				     std::plus<double>(),
				     fc::diff_squared<float,double>()
				     );

  double reference = std::inner_product(_data.data(),
					_data.data() + _data.num_elements(),
					_data.data(),
					0.);
  
  double value = std::sqrt(l2norm)/std::sqrt(reference);

  return value;
}


#endif /* _TEST_UTILS_H_ */
