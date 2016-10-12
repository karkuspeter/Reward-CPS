classdef ToyCannonSimulator3D < handle
    % Class for toy cannon problem
    %   2D cannon with random hills
    
    %TODO dont return hills.. and no need for reward function
    
    properties
        angleNoise
        s_bounds
        hill_count
        hill
        r_func
        PrintOn;
        cannon_h; %height of cannon
        step; %step size for simluation
        %hill_func;
    end
    
    methods
      function obj = ToyCannonSimulator3D()
         
         obj.angleNoise = 1/180*pi;
         obj.s_bounds = [-12, 12; -12, 12];
         obj.hill_count = 4;
         obj.hill = struct('c1', [-9 -4 4 9], ...
                           'c2', [0 -5 0 5], ...
                           'h', [.3 .3 .4 .2], ...
                           'scale', [4 4 8 16]);
         obj.step = 0.1;              

         %[obj.x1, obj.x2] = ndgrid(linspace(obj.s_bounds(1,1)*2,obj.s_bounds(1,2)*2, obj.representers), ...
         %    linspace(obj.s_bounds(2,1)*2,obj.s_bounds(2,2)*2, obj.representers));
         %obj.y = arrayfun(obj.HillValue, obj.x1, obj.x2);
         %obj.r_func = @(a,v,s,hillats,xres,yres)(4-sqrt( (xres-s).^2 + (yres - hillats)^2));
         %obj.r_func = @(a,v,s,hillats,xres,yres)(4-sqrt( (xres-s).^2 )  - 1.*v.^2); %- 0.1*a.^2
         obj.r_func = @(a_vert, a_hor, v, s1, s2, hillats, xres1, xres2, yres)...
               (-(xres1-s1).^2 -(xres2-s2).^2  - 1.*v.^2 - 3*a_vert.^2); %- 0.1*a.^2
        
         % reward: eucladian distance from target on the hill,
         % +4 to address 0 mean
         % could be x difference only, and add penalty for higher angle?
         obj.cannon_h = obj.HillValue(0, 0);
         obj.PrintOn = false;
      end

      function PlotEnv(obj)
          [x1, x2] = ndgrid(linspace(obj.s_bounds(1,1)*2,obj.s_bounds(1,2)*2, 100), ...
              linspace(obj.s_bounds(2,1)*2,obj.s_bounds(2,2)*2, 100));
          y = arrayfun(@obj.HillValue, x1, x2);
          mesh(x1, x2, y);
      end
      
      function y = HillValue(obj, x1, x2)
          y  = zeros(size(x1));  %hill values
          %for i=1:length(x)
          %    y(i) = obj.hill_func(x(i));
          %end
          for i = 1:length(obj.hill.c1)
             y = y + obj.hill.h(i) * exp(-((x1-obj.hill.c1(i)).^2 + (x2-obj.hill.c2(i)).^2)/obj.hill.scale(i));
          end
      end
      
      function [r, result] = Simulate(obj, s1, s2, angle_v, angle_h, v, noise)
          % reasonable input: s=3, angle=.3, v = 1;
          
          r = zeros(size(s1,1),1);
          result = zeros(size(s1,1),3);
          
          for i=1:size(s1,1)
              if nargin < 7
                  noise = obj.angleNoise;
              end

              th_v = angle_v(i,:) + randn(1)*noise;
              th_h = angle_h(i,:);
    
              traj = zeros(200, 4);  %x1, x2, ball_h, hill_h
              traj(1,:) = [0,0,obj.cannon_h,obj.cannon_h];
              step2d = [cos(th_h) sin(th_h)]*obj.step;              
              tan_th_v = tan(th_v);
              cos_th_v = cos(th_v);
              vert_acc = - 0.981 * 0.5/((v(i,:)*cos_th_v)^2);
              coord = [0 0];
              
              j = 2;
              while (true)
                  coord = traj(j-1, 1:2) + step2d;
                  d = (j-1)*obj.step;
                  ball_h = obj.cannon_h + d * tan_th_v +  d^2 * vert_acc; 
                  hill_h = obj.HillValue(coord(1), coord(2));
                  traj(j,:) = [coord, ball_h, hill_h];
                  
                  % break if ground hit
                  if ball_h < hill_h
                      break
                  end
                  % break if limit reached
                  if coord(1) < obj.s_bounds(1, 1)*2 || coord(1) > obj.s_bounds(1, 2)*2 ||...
                        coord(2) < obj.s_bounds(2, 1)*2 || coord(2) > obj.s_bounds(2, 2)*2   
                    break;
                  end
                  j = j+1;
              end
              traj = traj(1:j,:);
              
              hillats = obj.HillValue(s1(i,:), s2(i,:));
              r(i,:) = obj.r_func(th_v, th_h, v, s1(i,:), s2(i,:), hillats, coord(1), coord(2), ball_h);
              result(i,:) = [hillats,coord(1),coord(2)];

              %print
              if(obj.PrintOn)
                  %ixLand = find(bsxfun(@lt, yproj, obj.y));  % find all x where yproj < y
                  figure(1);
                  obj.PlotEnv();
                  hold on, plot3(traj(:,1), traj(:,2), traj(:,3), 'r--')
                  hold on, scatter3(s1(i,:), s2(i,:), hillats, 'ko')
              end
          end
      end
      
    end
    
end