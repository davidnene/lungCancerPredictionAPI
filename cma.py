def cma(cm):
  """
  Confusion Matrix Analyse
  """
  recall    = []
  precision = []
  f1        = []
  for i in range(cm.shape[0]):
    recall.append(cm[i][i]/cm.sum(axis=0)[i])
    precision.append(cm[i][i]/cm.sum(axis=1)[i])
    f1.append(2 * (((cm[i][i]/cm.sum(axis=0)[i]) * (cm[i][i]/cm.sum(axis=1)[i]))/((cm[i][i]/cm.sum(axis=0)[i]) + (cm[i][i]/cm.sum(axis=1)[i]))))

  recall    = [x for x in recall if str(x) != 'nan']
  precision = [x for x in precision if str(x) != 'nan']
  f1        = [x for x in f1 if str(x) != 'nan']

  return [sum(recall)/len(recall), sum(precision)/len(precision), sum(f1)/len(f1)]