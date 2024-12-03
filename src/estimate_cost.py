import re, io, sys, json
import types
from calflops import calculate_flops

def parse_params(param, unit):
  if unit == 'K':
    unit = 1e3
  elif unit == 'M':
    unit = 1e6
  elif unit == 'G':
    unit = 1e9
  return int(param * unit)

def parse_macs(macs, unit):
  if unit == 'MMACs':
    unit = 1e6
  elif unit == 'KMACs':
    unit = 1e3
  elif unit == 'GMACs':
    unit = 1e9
  elif unit == 'MACs':
    unit = 1
  return int(macs * unit)

def parse_flops(macs, unit):
  if unit == 'MFLOPS':
    unit = 1e6
  elif unit == 'KFLOPS':
    unit = 1e3
  elif unit == 'GFLOPS':
    unit = 1e9
  elif unit == 'FLOPS':
    unit = 1
  return int(macs * unit)

def extract_params_inline(line):
  params = macs = flops = 0
  if '(' in line and ':' in line: #there is a module name:
    #find second parenthesis, and remove last parenth
    line = line[line.find('(', line.find('(', 0) + 1) + 1: -1]
  params = line.split('Params')[0].split('=')[0].split()
  params = float(params[0]) if len(params) == 1 else parse_params(float(params[0]), params[1])
  macs = line.split(',')[1].split('=')[0].split()
  macs = float(macs[0]) if len(macs) == 1 else parse_macs(float(macs[0]), macs[1])
  flops = line.split(',')[2].split('=')[0].split()
  flops = float(flops[0]) if len(flops) == 1 else parse_flops(float(flops[0]), flops[1])
  bitwidth = re.findall(r'(\d+)bit', line)
  bitwidth = None if not bitwidth else int(bitwidth[0])

  return params, macs, flops, bitwidth
  
def parse_model_structure_re(lines, max_indent = -1):
    """ 
    Reads a list of strings (output of a FLOPS estimator) and creates a tree structure 
    representing the model's modules, with direct links to parent and children.
    """
    start = 0
    line = lines[0]
    root_parent_module_name = line.split('(')[0]

    params, macs, flops, bitwidth = extract_params_inline(lines[1])
    depth = cost = 0
    root = {
        "name": root_parent_module_name,
        "params": params,
        "macs": macs,
        "flops": flops,
        "bitwidth": None,
        "depth": depth,
        "cost": cost,
        "parent": None,
        "children": []
    }

    parsed_model_dict = root  # The root of the tree
    current_path = [root]

    i = 0
    while i + 1 < len(lines):
        i += 1
        line = lines[i]
        params = macs = flops = None

        depth = len(line) - len(line.lstrip())
        depth = depth // 2

        if '(' in line and ':' in line:  # if same line contains module name
            module_name = line[line.find('(', 0) + 1: line.find(')', 0)]
            if 'Params' in line and 'MACs' in line:  # if info and module name on same line
                params, macs, flops, bitwidth = extract_params_inline(line)
            elif 'Params' in lines[i + 1] and 'Params' not in line:  # if info is next line
                params, macs, flops, bitwidth = extract_params_inline(lines[i + 1])
                i += 1
            elif 'Params' not in line and 'Params' not in lines[i + 1]:
                continue

            # Create a new module node
            new_module = {
                "name": module_name,
                "params": params,
                "macs": macs,
                "flops": flops,
                "bitwidth": bitwidth,
                "cost": 0,
                "parent": None,
                "children": []
            }

            # Adjust the current path based on depth
            while len(current_path) > depth:
                current_path.pop()

            # Set the parent-child relationship
            parent_module = current_path[-1]
            new_module["parent"] = parent_module
            parent_module["children"].append(new_module)

            # Add the new module to the current path
            current_path.append(new_module)

    return parsed_model_dict

def return_cost_function_given_name(name):
  if isinstance(name, str):
    if name.upper() == 'ACE':
      return lambda macs,bit1,bit2 : macs * bit1 * bit2
  elif isinstance(name, types.LambdaType):
    return name
  else:
    raise TypeError(f"Error: no default cost function entitled {name}. Try ACE or review documentation.")
  
def update_cost_tree(module_dict, f, trace_quantization = True,  default_bitwidth = 32):
    # Check if the current item has quantizers as direct children
    # Dict format {parent_module_name: [params, macs, flops, bitwidth, depth, {}]}
    f = return_cost_function_given_name(f)
    params_order, macs_order, flops_order = 0, 1, 2
    bitwidth_order, depth_order, sub_dict_order = 3, 4, 6
    cost_order = 5
    if isinstance(module_dict, dict):
        macs, flops, bitwidth = module_dict['macs'], module_dict['flops'], module_dict['bitwidth']
        precision_input = precision_weight = default_bitwidth

        for child in module_dict['children']:
          if child['name'] == '_input_quantizer':
            precision_input = child['bitwidth']
            #nested quantizer bug from calflops
            for child_2 in child['children']:
              if child_2['name'] == '_weight_quantizer':
                precision_weight = child_2['bitwidth']
            if not trace_quantization:
                module_dict["children"].pop(module_dict["children"].index(child))
                
          if child['name'] == '_weight_quantizer':
            precision_input = child['bitwidth']
            #nested quantizer bug from calflops
            for child_2 in child['children']:
              if child_2['name'] == '_input_quantizer':
                precision_weight = child_2['bitwidth']
            if not trace_quantization:
              module_dict["children"].pop(module_dict["children"].index(child))

        curr_cost = f(macs, precision_input, precision_weight)
        unq_cost = f(macs, default_bitwidth, default_bitwidth)
        module_dict['cost'] = curr_cost

      # Recursively process child modules
        if module_dict['children'] != []:
          for child in module_dict['children']:
            update_cost_tree(child, f, trace_quantization)
        # update parent cost
        if curr_cost < unq_cost:
          # print(f"Quantization {precision_input, precision_weight} Cost Diff {unq_cost - curr_cost}")
          diff = unq_cost - curr_cost
          parent = module_dict['parent']
          while parent != None:
            parent['cost'] -= diff
            parent = parent['parent']


def estimate_cost(model, input_shape, f, trace_quantization = True, 
                  default_precision = 32, model_name = "MultiTaskSwin"):

  buffer = io.StringIO()
  sys.stdout = buffer
  flops, macs, params = calculate_flops(model=model, 
                                    input_shape=(1, 3, 224, 224),
                                    output_as_string=True,
                                    output_precision=4)
  sys.stdout = sys.__stdout__
  output_str = buffer.getvalue()
  buffer.close()
  output_str = output_str.splitlines()
  output_str = [line for line in output_str if not line.strip().startswith("Warning")]
  model_info = parse_model_structure_re(output_str)
  start_index = next((i for i, item in enumerate(output_str) if model_name in item), None)
  if start_index is not None:
      output_str = output_str[start_index:]
  else:
      print(f"No {model_name} found in Model Description.")
      output_str = []

  update_cost_tree(model_info, f, trace_quantization)
  total_model_cost = model_info['cost']
  print(json.dumps(model_info, indent=2))
  print(f"GMACS = {macs / 1e9} and params = {params/1e6} M")
  return model_info, total_model_cost