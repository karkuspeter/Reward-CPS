function [ params ] = ProcessParams( defaults,  input_params)
%overright params that are set in input_params

    params = defaults;
    fields = fieldnames(params);
    for i=1:numel(fields)
        if isfield(input_params, fields{i})
            params.(fields{i}) = input_params.(fields{i});
        end
    end

end

