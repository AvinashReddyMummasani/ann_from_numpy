import numpy as np
import tqdm
class Neural_network:
    def __init__(self,dim,activations,loss,learning_rate):
        self.learning_rate=learning_rate
        self.loss=loss
        self.activations_list=activations
        dim=dim
        self.dim=dim
        layers=len(dim)-1
        self.layers=layers
        weights={}
        bais={}
        for i in range(1,layers+1):
            weights['w'+str(i)]=np.random.rand(dim[i-1],dim[i])
            bais['b'+str(i)]=np.random.rand(dim[i],1)
        self.weights=weights
        self.bais=bais
    
    def loss_value(self,y,y_hat):
        if self.loss=='mse':
            return (y-y_hat)**2
        if self.loss=='bce':
            return (-y*np.log(y_hat))-(1-y)*np.log(1-y_hat)
    
    
    def sigmoid(self,i):
        return (1/(1+np.exp(-i)))
    
    def dif_relu(self,layer,outputs):
        A=[]
        for i in outputs[f"o{layer}"][0]:
            if i>0:
                A.append(1)
            else:
                A.append(0)
        A=np.array([A])
        return A
    
    def dif_sigmoid(self,layer,outputs):
        A=[]
        for i in outputs[f"o{layer}"][0]:
            sigma=i
            A.append(sigma*(1-sigma))

        return np.array([A])
    
    def dif_linear(self,layer,outputs):
        A=[]
        for i in outputs[f"o{layer}"][0]:
            A.append(1)

        return np.array([A])
    
    
    def dif_loss(self,y,y_hat):
        if self.loss=='mse':
            return np.array([[-2*(y-y_hat)]])
        if self.loss=='bce':
            return (y_hat - y) / (y_hat * (1 - y_hat))





    def dif_activation(self,layer,outputs):
        if self.activations_list[layer-1]=='relu':
            return self.dif_relu(layer,outputs)
        if self.activations_list[layer-1]=='sigmoid':
            return self.dif_sigmoid(layer,outputs)
        if self.activations_list[layer-1]=='linear':
            return self.dif_linear(layer,outputs)
    
    def relu(self,inp):
        A=[]
        for i in inp[0]:
            if i>=0:
                A.append(i)
            else:
                A.append(0)
        return np.array([A])
    
    def sigmoid_activation(self,inp):
        A=[]
        for i in inp[0]:
            A.append(self.sigmoid(i))
        return np.array([A])
    

    
    def activation(self,layer,inp):
        if self.activations_list[layer-1]=='relu':
            return self.relu(inp)
        if self.activations_list[layer-1]=='sigmoid':
            return self.sigmoid_activation(inp)
        if self.activations_list[layer-1]=='linear':
            return inp
        
    

    def forward_propagation(self,x):
        outputs={}
        outputs['o0']=[x]
        weights=self.weights
        bais=self.bais
        for i in range(1,self.layers+1):
            z = np.dot(outputs[f'o{i-1}'], weights[f'w{i}']) + bais[f"b{i}"].T
            outputs[f'o{i}'] = self.activation(i, z)

        return outputs['o'+str(self.layers)][0][0],outputs
    
    def deepcopy(self,d):
        temp={}
        for i in d:
            temp[i]=d[i].copy()
        return temp
    
    def back_propagation(self,y_hat,y,outputs):
        weights=self.weights
        bais=self.bais
        learning_rate=self.learning_rate
        weights_=self.deepcopy(weights)
        bais_=self.deepcopy(bais)
        n=len(outputs)-1
        back=self.dif_loss(y,y_hat)
        back=back*self.dif_activation(n,outputs)
        for j in range(n,0,-1):
            weights['w'+str(j)]=weights['w'+str(j)]-learning_rate*(np.dot(np.array(outputs['o'+str(j-1)]).T,back))
            bais[f"b{j}"]=bais[f"b{j}"]-learning_rate*(back.T)
            if j != 1:
                dif_a = self.dif_activation(j-1, outputs)   
                back = (back @ weights_[f"w{j}"].T) * dif_a   

        del weights_
        del bais_
        return
    
    def fit(self,X_train,y_train,epochs=5):

        for i in range(epochs):
            loss=[]
            print(f"epoch{i+1} :",end=" ")
            for j in tqdm.tqdm(range(X_train.shape[0])):
                ind=np.random.randint(0,X_train.shape[0])
                x=np.array([np.float64(i) for i in X_train[ind]])
                pred=self.forward_propagation(x)
                y_hat=pred[0]
                outputs=pred[1]
                loss.append((y_train[ind]-y_hat)**2)
                self.back_propagation(y_hat,y_train[ind],outputs)
            print("  loss:",np.array(loss).mean())
    def predict(self,X_test):
        y_pred=[]
        for i in X_test:
            y_pred.append(self.forward_propagation(i)[0])
        return np.array(y_pred)

