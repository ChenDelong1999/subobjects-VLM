

def get_params_count_summary(model, max_name_len: int = 96):
  padding = 64
  
  params = [(name[:max_name_len], p.numel(), str(tuple(p.shape)), p.requires_grad) for name, p in model.named_parameters()]
  total_trainable_params = sum([x[1] for x in params if x[-1]])
  total_nontrainable_params = sum([x[1] for x in params if not x[-1]])
  
  param_counts_text = ''
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Module":<{max_name_len}} | {"Trainable":<8} | {"Shape":>20} | {"Param Count":>13} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  
  for name, param_count, shape, trainable in params:
      truncated_name = name[:max_name_len]  # Truncate the name if it's too long
      param_counts_text += f'| {truncated_name:<{max_name_len}} | {"True" if trainable else "False":<8} | {shape:>20} | {param_count:>13,} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Total trainable params":<{max_name_len}} | {"":<8} | {"":<20} | {total_trainable_params:>13,} |\n'
  param_counts_text += f'| {"Total non-trainable params":<{max_name_len}} | {"":<8} | {"":<20} | {total_nontrainable_params:>13,} |\n'
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  
  return param_counts_text