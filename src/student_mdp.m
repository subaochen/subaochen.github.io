gamma=1.0 # gamma
R=[-2;-2;-2;10;1;-1;0] # immediately reward
P=[0 0.5 0 0 0 0.5 0; # status transition matrix
0 0 0.8 0 0 0 0.2;
0 0 0 0.6 0.4 0 0;
0 0 0 0 0 0 1.0;
0.2 0.4 0.4 0 0 0 0;
0.1 0 0 0 0 0.9 0;
0 0 0 0 0 0 0]
V=inv(eye(7)-gamma*P)*R # status value