import numpy as np

class SVM:
  def __init__(self, C):
    self.C = C
    self.w = None
    self.b = None
    self.X = None
    self.y = None
    self.n = 0
    self.d = 0

  def decision_function(self, X):
    return np.dot(X, self.w) + self.b

  def cost(self, margin):
    hinge_loss = np.maximum(0, 1 - margin)
    return 0.5 * np.dot(self.w, self.w) + self.C * np.mean(hinge_loss)

  def margin(self, X, y):
    return y * self.decision_function(X)

  def fit(self, X, y, lr=1e-3, epochs=500):
    self.n, self.d = X.shape
    self.w = np.zeros(self.d)
    self.b = 0

    for _ in range(epochs):
      margin = self.margin(X, y)

      hinge_loss_grad = np.where(margin < 1, -self.C * y, 0)
      d_w = self.w + np.dot(hinge_loss_grad, X)
      self.w -= lr * d_w

      d_b = -self.C * np.sum(y[margin < 1])
      self.b -= lr * d_b

  def predict(self, X):
    return np.sign(self.decision_function(X))