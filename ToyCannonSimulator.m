classdef ToyCannonSimulator < handle
    % Class for toy cannon problem
    %   2D cannon with random hills
    
    %TODO dont return hills.. and no need for reward function
    
    properties
        angleNoise
        s_bounds
        hill_count
        representers
        hill
        x  %positions on ground, cached
        y  %hill values for each x
        r_func
        PrintOn;
        hill_func;
    end
    
    methods
      function obj = ToyCannonSimulator()
         
         obj.angleNoise = 1/180*pi;
         obj.s_bounds = [0, 12];
         obj.hill_count = 4;
         obj.hill = struct('c', [3 5 6 7], ...
                           'h', [.3 .3 .4 .2], ...
                           'scale', [.5 .5 .1 2]);
         obj.representers = 1001;
                       
         % function describing hills, not used for now
         single_hill_func = @(x, h, c, scale)(h * exp(-(x-c).^2/scale));
         obj.hill_func = @(x)(sum(single_hill_func(x, obj.hill.h, obj.hill.c, obj.hill.scale)));

         obj.x = linspace(obj.s_bounds(1),obj.s_bounds(2)*2, obj.representers);
         obj.y = obj.HillValue(obj.x);
         %obj.r_func = @(a,v,s,hillats,xres,yres)(4-sqrt( (xres-s).^2 + (yres - hillats)^2));
         %obj.r_func = @(a,v,s,hillats,xres,yres)(4-sqrt( (xres-s).^2 )  - 1.*v.^2); %- 0.1*a.^2
         obj.r_func = @(a,v,s,hillats,xres,yres)(4-sqrt( (xres-s).^2 )  - 1.*v.^2 - 0*a.^2); %- 0.1*a.^2
        
         % reward: eucladian distance from target on the hill,
         % +4 to address 0 mean
         % could be x difference only, and add penalty for higher angle?
         
         obj.PrintOn = false;
      end

      function PlotEnv(obj)
          x = obj.s_bounds(1):0.1:obj.s_bounds(2);
          plot(x, obj.HillValue(x))
      end
      
      function y = HillValue(obj, x)
          y  = zeros(size(x));  %hill values
          %for i=1:length(x)
          %    y(i) = obj.hill_func(x(i));
          %end
          for i = 1:length(obj.hill.c)
             y = y + obj.hill.h(i) * exp(-(x-obj.hill.c(i)).^2/obj.hill.scale(i));
          end
      end
      
      function [r, result] = Simulate(obj, s, angle, v, noise)
          % reasonable input: s=3, angle=.3, v = 1;
          
          r = zeros(size(s,1),1);
          result = zeros(size(s,1),3);
          
          for i=1:size(s,1)
              if nargin < 5
                  noise = obj.angleNoise;
              end

              th = angle(i,:) + randn(1)*noise;

              yproj = obj.y(1) + obj.x.*tan(th) - 9.81/100 * obj.x.^2/2/(v(i,:)*cos(th))^2; %y of ball trajectory
              %f_yproj = @(x)(x.*tan(th) - 9.81/100 * x.^2/2/(v*cos(th))^2); %y of ball trajectory

              % find intersection with a general search, too slow
              %[ixLand, yres] = intersections(obj.x, obj.y, obj.x, yproj);

              ixLand = find(bsxfun(@lt, yproj(2:end), obj.y(2:end)),1);  % find all x where yproj < y
              if(length(ixLand) < 1)
                  xres = obj.x(end);
                  yres = obj.y(end);
              else
                   xres = obj.x(ixLand);
                  yres = obj.y(ixLand);
              end

              %evaluate
              hillats = obj.HillValue(s(i,:));
              r(i,:) = obj.r_func(angle(i,:),v(i,:),s(i,:),hillats,xres,yres);
              result(i,:) = [hillats,xres,yres];

              %print
              if(obj.PrintOn)
                  %ixLand = find(bsxfun(@lt, yproj, obj.y));  % find all x where yproj < y
                  figure(1);
                  obj.PlotEnv();
                  hold on, plot(obj.x(1:ixLand), yproj(1:ixLand), 'r--')
                  hold on, plot(s(i,:), ycontext, 'ko')
              end
          end
      end
      
    end
    
end