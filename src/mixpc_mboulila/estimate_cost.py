import re, io, sys, json
from ptflops import get_model_complexity_info

def parse_params(param, unit):
  if unit == 'k':
    unit = 1e3
  elif unit == 'M':
    unit = 1e6
  elif unit == 'G':
    unit = 1e9
  return int(param * unit)

def parse_macs(macs, unit):
  if unit == 'MMac':
    unit = 1e6
  elif unit == 'KMac':
    unit = 1e3
  elif unit == 'GMac':
    unit = 1e9
  elif unit == 'Mac':
    unit = 1
  return int(macs * unit)
  
def parse_model_structure_re(lines, max_indent = -1):
    #TODO: remove
    # print("first 10 lines:")
    # print(lines[:10])

    parent = lines[0].split('(')[0]
    params, macs, depth, cost = 0, 0, 0, 0
    pattern = r'(\d+\.?\d*\s*[kKM]*[M]?Mac)\s*,\s*(\d+\.?\d*%)\s*MACs|(\d+\.?\d*\s*[kKM]*)\s*,\s*(\d+\.?\d*%)\s*Params'
    matches = re.findall(pattern, lines[1])
    # print(matches)
    params = matches[0][2].split()
    macs = matches[1][0].split()
    params = float(params[0]) if len(params) == 1 else parse_params(float(params[0]), params[1])
    macs = float(macs[0]) if len(macs) == 1 else parse_macs(float(macs[0]), macs[1])
    model_dict = {parent: [params, macs, depth, cost, {}]}
    current_path = [parent]

    i = 1
    while i + 1 < len(lines):
        i += 1
        line = lines[i]
        depth = len(line) - len(line.lstrip())
        depth = depth // 2
        name_pattern = r'^\s*\(([\w_]+)\):'
        name_matches = re.findall(name_pattern, line)
        print(line)
        if len(name_matches) > 0:
          module_name = name_matches[0]
          params, macs = 0, 0
          next_line = lines[i+1]
          #case 1: module and parameters are on the same line
          if 'Params' not in line and 'Params' not in next_line:
              continue
          else:
            #case 2: module first line, parameters second line
            if 'Params' in next_line and 'Params' not in line:
              line = next_line
              i += 1
            matches = re.findall(pattern, line)
            # print(module_name)
            # print(matches)
            # print(line)
            # print(matches)
            params = matches[0][2].split()
            # print(params)
            params = float(params[0]) if len(params) == 1 else parse_params(float(params[0]), params[1])
            params_percentage = matches[0][3]
            macs = matches[1][0].split()
            # print(macs)
            macs = float(macs[0]) if len(macs) == 1 else parse_macs(float(macs[0]), macs[1])
            macs_percentage = matches[1][1]
            # print(f"MACs: {macs}, {macs_percentage}")
            # print(f"Params: {params}, {params_percentage}")
            precision_find = re.findall(r'(\d+)bit', line)
            precision = 0 if not precision_find else int(precision_find[0])
            cost = precision

            # Adjust the current path based on depth
            while len(current_path) > depth:
                current_path.pop()

            current_dict = model_dict
            for path_part in current_path:
                current_dict = current_dict[path_part][-1]

            current_dict[module_name] = [params, macs, depth, cost, {}]
            current_path.append(module_name)

    return model_dict

def update_cost_with_precision(module_dict, f, trace_quantization = True):
    # Check if the current item has quantizers as direct children
    if isinstance(module_dict, dict):
        for key, value in module_dict.items():
            if isinstance(value, list):
                #params, macs, cost, depth, sub-modules in order
                macs = value[1]
                cost = value[2]
                precision_input = precision_weight = 32  # Default precision if quantizers are not found
                if '_input_quantizer' in value[4]:
                    precision_input = value[4]['_input_quantizer'][2]
                if '_weight_quantizer' in value[4]:
                    precision_weight = value[4]['_weight_quantizer'][2]

                value[3] = f(macs, precision_input, precision_weight)
                if not trace_quantization:
                  value[4].pop('_input_quantizer', None)
                  value[4].pop('_weight_quantizer', None)

            # Recursively process child modules
            update_cost_with_precision(value[4], f, trace_quantization)

#TODO: change name
def return_cost_with_precision(module_dict, f, trace_quantization = True, default_precision = 32):
    # Check if the current item has quantizers as direct children
    if isinstance(module_dict, dict):
        total_cost = 0
        for key, value in module_dict.items():
            if isinstance(value, list) and len(value) > 0:
                #params, macs, cost, depth, sub-modules in order
                macs = value[1]
                # print(key, macs)
                #case 1 quantized layer
                if '_input_quantizer' in value[4]:
                    # print(key, value[4])
                    precision_input = value[4]['_input_quantizer'][3]
                    precision_weight = value[4]['_weight_quantizer'][3] if '_weight_quantizer' in value[4] else 8
                    cost = f(macs, precision_input, precision_weight)
                    if not trace_quantization:
                      value[4].pop('_input_quantizer', None)
                      value[4].pop('_weight_quantizer', None)
                elif len(value[4]) == 0: #case 2  non quantized layer
                    cost = f(macs, default_precision, default_precision)
                else: #case 3 a sub-model
                    cost = return_cost_with_precision(value[4], f, trace_quantization, default_precision)
                # print(cost)
                total_cost += cost
        value[3] = total_cost
        return total_cost
    
def estimate_cost(model, input_shape, f, trace_quantization = True, default_precision = 32):

  buffer = io.StringIO()
  sys.stdout = buffer
  macs, params = get_model_complexity_info(model, input_shape, as_strings=False,
                                              print_per_layer_stat=True, verbose=True,
                                              ost = sys.stdout)
  sys.stdout = sys.__stdout__
  output_str = buffer.getvalue()
  buffer.close()
  output_str = output_str.splitlines()
  output_str = [line for line in output_str if not line.strip().startswith("Warning")]
  model_info = parse_model_structure_re(output_str)
  total_model_cost = return_cost_with_precision(model_info, f, True)
  print(json.dumps(model_info, indent=2))
  print(f"GMACS = {macs / 1e9} and params = {params/1e6} M")
  return model_info, total_model_cost