#ifndef rho
#define rho ${'{0:6.1f}'.format(rho_var)} /* fluid density (kg/m^3) */
#endif

% for bc_name in bcs.keys():
    % if bc_name == inlet_name:
        <% continue %>
    % endif

/* resistance (Pa * s / m^3 ) ${bc_name} */
#ifndef R_${bc_name}
#define R_${bc_name} ${'{0:.8e}'.format(bcs[bc_name]["R"])}
#endif

/* venous capacitance (m^3 / Pa) ${bc_name} */
#ifndef C_${bc_name}
#define C_${bc_name} ${'{0:.8e}'.format(bcs[bc_name]["C"])}
#endif

/* initial pressure (Pa) ${bc_name} */
#ifndef p_${bc_name}
#define p_${bc_name} ${'{0:6d}'.format(bcs[bc_name]["P"])}
#endif

% endfor
