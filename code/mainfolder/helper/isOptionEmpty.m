function bool = isOptionEmpty( op, field )
%ISOPTIONEMPTY Check if a field in the struct is non-existent or empty ([])
%
bool = ~( isfield(op, field) && ~isempty(op.(field))  );

end

