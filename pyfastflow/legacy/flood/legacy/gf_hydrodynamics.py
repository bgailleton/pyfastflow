"""
Hydrodynamic computation kernels for Flood shallow water flow.

This module implements the core hydrodynamic algorithms for 2D shallow water
flow simulation using GPU-accelerated Taichi kernels. It provides methods for
discharge diffusion and Manning's equation-based flow depth updates.

Key algorithms:
- Discharge diffusion for multiple flow path simulation
- Manning's equation for flow resistance and depth updates
- Integration with FastFlow's flow routing system

Based on methods from Gailleton et al. 2024 for efficient
shallow water flow approximation, adapted to GPU (Gailleton et al., in prep).

Author: B.G.
"""

import taichi as ti

import pyfastflow.flow as flow

from ... import constants as cte


@ti.kernel
def add_P_to_h(prec:ti.template(), h:ti.template(), dt:ti.f32):
    for i in h:
        h[i]+=prec[i]*dt

@ti.kernel
def _set_outlet_to(h:ti.template(), val:ti.f32):
    for i in h:
        # Skip boundary cells
        if flow.neighbourer_flat.can_leave_domain(i):
            h[i] = val


@ti.kernel
def diffuse_Q_constant_prec(z: ti.template(), Q: ti.template(), temp: ti.template()):
    """
    Diffuse discharge field to simulate multiple flow paths.

    Redistributes discharge from each cell to its neighbors based on slope
    gradients, creating a more realistic multiple flow direction pattern
    from the original single flow direction (SFD) routing.

    The method:
    1. Initializes precipitation input for each cell
    2. Computes slope-weighted diffusion to neighbors
    3. Redistributes discharge proportionally to slope gradients

    Args:
            z (ti.template): Combined surface elevation field (topography + water depth)
            Q (ti.template): Discharge field to diffuse
            temp (ti.template): Temporary field for intermediate calculations

    Author: B.G.
    """

    # Initialize precipitation input and handle boundary conditions
    for i in Q:
        temp[i] = cte.PREC * cte.DX * cte.DX  # Add precipitation as volume input

    # Diffuse discharge based on slope gradients
    for i in z:
        # Skip boundary cells
        if flow.neighbourer_flat.can_leave_domain(i):
            continue

        # Calculate total slope gradient sum for normalization
        sums = 0.0
        for k in range(4):  # Check all 4 neighbors
            j = flow.neighbourer_flat.neighbour(i, k)
            sums += ti.max(0.0, (((z[i]) - (z[j])) / cte.DX) if j != -1 else 0.0)

        # Skip cells with no downslope neighbors
        if sums == 0.0:
            continue

        # Distribute discharge proportionally to slope gradients
        for k in range(4):
            j = flow.neighbourer_flat.neighbour(i, k)
            tS = ti.max(0.0, (((z[i]) - (z[j])) / cte.DX) if j != -1 else 0.0)
            ti.atomic_add(temp[j], tS / sums * Q[i])  # Add proportional discharge

    # Update discharge field with diffused values
    for i in Q:
        Q[i] = temp[i]


@ti.kernel
def graphflood_core_cte_mannings(
    h: ti.template(),
    z: ti.template(),
    dh: ti.template(),
    rec: ti.template(),
    Q: ti.template(),
):
    """
    Core shallow water flow computation using Manning's equation.

    Implements the main hydrodynamic computation for 2D shallow water flow
    using Manning's equation for flow resistance. Updates water depth based
    on discharge input and outflow capacity.

    The method:
    1. Computes local slope from flow receivers
    2. Calculates outflow capacity using Manning's equation
    3. Updates water depth based on discharge balance
    4. Ensures non-negative depth values
    5. Maintains separate water depth field (bed elevation unchanged)

    Based on core methods from Gailleton et al. 2024.

    Args:
            h (ti.template): Flow depth field
            z (ti.template): Combined surface elevation field (topography + water depth)
            dh (ti.template): Depth change field for intermediate calculations
            rec (ti.template): Flow receiver field from flow routing
            Q (ti.template): Discharge field

    Author: B.G.
    """

    # Compute depth changes using Manning's equation
    for i in h:
        # Determine local slope
        tS = cte.EDGESW  # Use edge slope for boundary/sink cells
        if rec[i] != i:  # Interior cells with valid receivers
            tS = ti.max(
                ((z[i] + h[i]) - (z[rec[i]] + h[rec[i]])) / cte.DX, 1e-4
            )  # Slope to receiver (minimum 1e-4)

        # Calculate outflow capacity using Manning's equation
        # Q = (1/n) * A * R^(2/3) * S^(1/2), where R ≈ h for wide channels
        Qo = cte.DX * h[i] ** (5.0 / 3.0) / cte.MANNING * ti.math.sqrt(tS)

        # Update depth based on discharge balance (inflow - outflow)
        tdh = (Q[i] - Qo) / (cte.DX**2) * cte.DT_HYDRO  # Volume change per unit area

        # Apply depth change and ensure non-negative depths
        # h[i] += tdh
        if h[i] + tdh < 0:  # Prevent negative depths
            tdh = -h[i]  # Adjust change to reach zero depth
        dh[i] = tdh

    # Apply final water depth changes
    for i in h:
        h[i] += dh[i]  # Apply final depth change






# @ti.kernel
# def graphflood_diffuse_cte_P_cte_man(z: ti.template(), h:ti.template(), Q: ti.template(), temp: ti.template(), dh: ti.template(), srecs: ti.template(), LM: ti.template(), temporal_filtering:cte.FLOAT_TYPE_TI):
#     """
#     NEXT STEPS::add a tag that propagate from local minimas and reroute from corrected receivers

#     Author: B.G.
#     """

#     # Initialize precipitation input and handle boundary conditions
#     for i in Q:
#         temp[i] = cte.PREC * cte.DX * cte.DX  # Add precipitation as volume input
#         dh[i] = 0.

#     # Diffuse discharge based on slope gradients
#     for i in z:
#         # Skip boundary cells
#         if flow.neighbourer_flat.can_leave_domain(i):
#             continue

#             # Calculate total slope gradient sum for normalization
#         sums = 0.0
#         msx  = 0.0
#         msy  = 0.0
#         isLM = True
#         for k in ti.static(range(4)):  # Check all 4 neighbors
#             j = flow.neighbourer_flat.neighbour(i, k)
#             ts = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j]))) if j != -1 else 0.0)
#             if ts>0:
#                 isLM = False

#          # Skip cells with no downslope neighbors
#         if isLM:
#             h[i] = z[srecs[i]] + h[srecs[i]] - z[i] + 1.
#             # continue
    
#         for k in ti.static(range(4)):  # Check all 4 neighbors
#             j = flow.neighbourer_flat.neighbour(i, k)
#             ts = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
#             sums += ts
#             msx = ti.max(ts, msx) if k == 1 or k == 2 else msx
#             msy = ti.max(ts, msy) if k == 0 or k == 3 else msy



#         if(sums == 0.):
#             ti.atomic_add(temp[srecs[i]], Q[i])
#         else:
#         # Distribute discharge proportionally to slope gradients
#             for k in range(4):
#                 j = flow.neighbourer_flat.neighbour(i, k)
#                 tS = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
#                 ti.atomic_add(temp[j], tS / sums * Q[i])  # Add proportional discharge

#         norms = ti.math.sqrt(ti.max(msx**2 + msy**2, 1e-3)) if isLM == False else 1e-4
#         if isLM:
#             h[i] -= 1.

#         Qo = cte.DX * h[i] ** (5.0 / 3.0) / cte.MANNING * ti.math.sqrt(norms)
#         dh[i] -= cte.DT_HYDRO * Qo/cte.DX**2

#     # Update discharge field with diffused values
#     for i in Q:
#         Q[i] = temp[i] * (1 - temporal_filtering ) + Q[i] * temporal_filtering
#         dh[i] += cte.DT_HYDRO * Q[i]/cte.DX**2
#         h[i] += dh[i]


# @ti.kernel
# def graphflood_diffuse_cte_P_cte_man(z: ti.template(), h:ti.template(), Q: ti.template(), temp: ti.template(), dh: ti.template(), srecs: ti.template(), LM: ti.template(), temporal_filtering:cte.FLOAT_TYPE_TI):
#     """
#     NEXT STEPS::add a tag that propagate from local minimas and reroute from corrected receivers

#     Author: B.G.
#     """

#     # Initialize precipitation input and handle boundary conditions
#     for i in Q:
#         temp[i] = cte.PREC * cte.DX * cte.DX  # Add precipitation as volume input
#         dh[i] = 0.

#     # Diffuse discharge based on slope gradients
#     for i in z:
#         # Skip boundary cells
#         if flow.neighbourer_flat.can_leave_domain(i):
#             continue

#             # Calculate total slope gradient sum for normalization
#         sums = 0.0
#         msx  = 0.0
#         msy  = 0.0
#         isLM = False


        
#         for k in ti.static(range(4)):  # Check all 4 neighbors
#             j = flow.neighbourer_flat.neighbour(i, k)
#             ts = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
#             sums += ts
#             msx = ti.max(ts, msx) if k == 1 or k == 2 else msx
#             msy = ti.max(ts, msy) if k == 0 or k == 3 else msy


#         # Skip cells with no downslope neighbors
#         if sums == 0.0:
#            isLM = True
#             # h[i] = mz - z[i] + 2e-3
#             # continue


            
        
#         if(isLM):
#             tN = 0
#             for k in range(4):
#                 j = flow.neighbourer_flat.neighbour(i, k)
#                 if(j != -1 and srecs[j] != i):
#                     tN += 1
#             for k in range(4):
#                 j = flow.neighbourer_flat.neighbour(i, k)
#                 if(j != -1 and srecs[j] != i):
#                     ti.atomic_add(temp[srecs[i]], Q[i]/tN)
#             # LM[srecs[i]] = True

#         else:
#             # Distribute discharge proportionally to slope gradients
#             for k in ti.static(range(4)):
#                 j = flow.neighbourer_flat.neighbour(i, k)
#                 tS = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
#                 ti.atomic_add(temp[j], tS / sums * Q[i])  # Add proportional discharge

#         norms = ti.math.sqrt(msx**2 + msy**2) if LM[i] == False else ti.max((z[i]+h[i] - z[srecs[i]] - h[srecs[i]])/cte.DX,1e-3)
#         Qo = cte.DX * h[i] ** (5.0 / 3.0) / cte.MANNING * ti.math.sqrt(norms)
#         dh[i] -= cte.DT_HYDRO * Qo/cte.DX**2

#     # Update discharge field with diffused values
#     for i in Q:
#         Q[i] = temp[i] * (1 - temporal_filtering ) + Q[i] * temporal_filtering
#         dh[i] += cte.DT_HYDRO * Q[i]/cte.DX**2
#         h[i] += dh[i]

# @ti.kernel
# def graphflood_diffuse_cte_P_cte_man(z: ti.template(), h:ti.template(), Q: ti.template(), temp: ti.template(), dh: ti.template(), srecs: ti.template(), LM: ti.template(), temporal_filtering:cte.FLOAT_TYPE_TI):
#     """
#     NEXT STEPS::add a tag that propagate from local minimas and reroute from corrected receivers

#     Author: B.G.
#     """

#     # Initialize precipitation input and handle boundary conditions
#     for i in Q:
#         temp[i] = cte.PREC * cte.DX * cte.DX  # Add precipitation as volume input
#         dh[i] = 0.

#     # Diffuse discharge based on slope gradients
#     for i in z:
#         # Skip boundary cells
#         if flow.neighbourer_flat.can_leave_domain(i):
#             continue

#             # Calculate total slope gradient sum for normalization
#         sums = 0.0
#         msx  = 0.0
#         msy  = 0.0
        
#         for k in range(4):  # Check all 4 neighbors
#             j = flow.neighbourer_flat.neighbour(i, k)
#             if(j == -1 or flow.neighbourer_flat.nodata(j)):
#                 continue
#             if(srecs[j] == i and LM[i]):
#                 continue
#             ts = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
#             sums += ts
#             msx = ti.max(ts, msx) if k == 1 or k == 2 else msx
#             msy = ti.max(ts, msy) if k == 0 or k == 3 else msy


#         # Skip cells with no downslope neighbors
#         if sums == 0.0:
#             LM[i] = True
#             # ti.atomic_add(temp[srecs[i]], Q[i])
#             # h[i] = mz - z[i] + 2e-3
#             # continue


            
        
        
#         if LM[i] and LM[srecs[i]] == False:
#             LM[srecs[i]] = True

#         if(LM[i] and sums == 0.):
#             ti.atomic_add(temp[srecs[i]], Q[i])
#         else:
#             # Distribute discharge proportionally to slope gradients
#             for k in range(4):
#                 j = flow.neighbourer_flat.neighbour(i, k)
#                 if(j == -1):
#                     continue
#                 if(srecs[j] == i and LM[i]):
#                     continue
#                 tS = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
#                 ti.atomic_add(temp[j], tS / sums * Q[i])  # Add proportional discharge

#         norms = ti.math.sqrt(msx**2 + msy**2) if (sums > 0.) else ti.max((z[i]+h[i] - z[srecs[i]] - h[srecs[i]])/cte.DX,1e-3)
#         Qo = cte.DX * h[i] ** (5.0 / 3.0) / cte.MANNING * ti.math.sqrt(norms)
#         dh[i] -= cte.DT_HYDRO * Qo/cte.DX**2

#     # Update discharge field with diffused values
#     for i in Q:
#         Q[i] = temp[i] * (1 - temporal_filtering ) + Q[i] * temporal_filtering
#         dh[i] += cte.DT_HYDRO * Q[i]/cte.DX**2
#         h[i] += dh[i]


# This test is with a damier like pattern to only fill some of the local minimas.
# @ti.kernel
# def graphflood_diffuse_cte_P_cte_man(z: ti.template(), h:ti.template(), Q: ti.template(), temp: ti.template(), dh: ti.template(), srecs: ti.template(), LM: ti.template(), temporal_filtering:cte.FLOAT_TYPE_TI):
#     """
#     NEXT STEPS::add a tag that propagate from local minimas and reroute from corrected receivers

#     Author: B.G.
#     """

#     # Initialize precipitation input and handle boundary conditions
#     for i in Q:
#         temp[i] = cte.PREC * cte.DX * cte.DX  # Add precipitation as volume input
#         dh[i] = 0.

#     # DOES NOT WORK IN TAICHI -> outter loop always the main parallel one
#     # ti.loop_config(serialize=True)
#     # for damier in range(2):

#     # Diffuse discharge based on slope gradients
#     for i in z:

#         if (i % 2) == 0:
#             continue


#         # Skip boundary cells
#         if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
#             continue

#             # Calculate total slope gradient sum for normalization
#         sums = 0.0
#         msx  = 0.0
#         msy  = 0.0
#         mz   = (z[i]+h[i])
#         tlm = True

#         for k in range(4):  # Check all 4 neighbors
#             j = flow.neighbourer_flat.neighbour(i, k)
#             if(j == -1):
#                 continue
            
#             mz  = ti.max(mz, (z[j]+h[j]))

#             if z[j]+h[j]<z[i]+h[i]:
#                 tlm = False
#                 break

#         # Skip cells with no downslope neighbors
#         if sums == 0.0 and tlm:
#             # LM[i] = True
#             # ti.atomic_add(temp[srecs[i]], Q[i])
#             h[i] = mz - z[i] + 5e-3
#             # continue

#         for k in range(4):  # Check all 4 neighbors
#             j = flow.neighbourer_flat.neighbour(i, k)
#             if(j == -1):
#                 continue

#             ts = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
#             sums += ts
#             msx = ti.max(ts, msx) if k == 1 or k == 2 else msx
#             msy = ti.max(ts, msy) if k == 0 or k == 3 else msy

        
   
#         # if LM[i] and LM[srecs[i]] == False:
#         #     LM[srecs[i]] = True

#         # if(LM[i] and sums == 0.):
#         #     ti.atomic_add(temp[srecs[i]], Q[i])
#         # else:
#         # Distribute discharge proportionally to slope gradients
#         for k in range(4):
#             j = flow.neighbourer_flat.neighbour(i, k)
#             if(j == -1):
#                 continue

#             tS = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
#             ti.atomic_add(temp[j], tS / sums * Q[i])  # Add proportional discharge

#         norms = ti.math.sqrt(msx**2 + msy**2) # if (sums > 0.) else ti.max((z[i]+h[i] - z[srecs[i]] - h[srecs[i]])/cte.DX,1e-3)
#         norms = ti.max(norms, 1e-6)
#         Qo = cte.DX * h[i] ** (5.0 / 3.0) / cte.MANNING * ti.math.sqrt(norms)
#         dh[i] -= cte.DT_HYDRO * Qo/cte.DX**2


#     # Diffuse discharge based on slope gradients
#     for i in z:

#         if (i % 2) == 1:
#             continue

#         # Skip boundary cells
#         if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
#             continue

#             # Calculate total slope gradient sum for normalization
#         sums = 0.0
#         msx  = 0.0
#         msy  = 0.0
#         mz   = (z[i]+h[i])
#         tlm = True
             
#         for k in range(4):  # Check all 4 neighbors
#             j = flow.neighbourer_flat.neighbour(i, k)
#             if(j == -1):
#                 continue
#             mz  = ti.max(mz, (z[j]+h[j]))
#             if z[j]+h[j]<z[i]+h[i]:
#                 tlm = False
#                 break

#         # Skip cells with no downslope neighbors
#         if sums == 0.0 or tlm:
#             # LM[i] = True
#             ti.atomic_add(temp[srecs[i]], Q[i])
#             # h[i] = mz - z[i] + 5e-3
#             # continue
#         else:
#             for k in range(4):  # Check all 4 neighbors
#                 j = flow.neighbourer_flat.neighbour(i, k)
#                 if(j == -1):
#                     continue

#                 ts = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
#                 sums += ts
#                 msx = ti.max(ts, msx) if k == 1 or k == 2 else msx
#                 msy = ti.max(ts, msy) if k == 0 or k == 3 else msy

            
       
#             # if LM[i] and LM[srecs[i]] == False:
#             #     LM[srecs[i]] = True

#             # if(LM[i] and sums == 0.):
#             #     ti.atomic_add(temp[srecs[i]], Q[i])
#             # else:
#             # Distribute discharge proportionally to slope gradients
#             for k in range(4):
#                 j = flow.neighbourer_flat.neighbour(i, k)
#                 if(j == -1):
#                     continue

#                 tS = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
#                 ti.atomic_add(temp[j], tS / sums * Q[i])  # Add proportional discharge

#         norms = ti.math.sqrt(msx**2 + msy**2) # if (sums > 0.) else ti.max((z[i]+h[i] - z[srecs[i]] - h[srecs[i]])/cte.DX,1e-3)
#         norms = ti.max(norms, 1e-6)
#         Qo = cte.DX * h[i] ** (5.0 / 3.0) / cte.MANNING * ti.math.sqrt(norms)
#         dh[i] -= cte.DT_HYDRO * Qo/cte.DX**2


#     # Update discharge field with diffused values
#     for i in Q:
#         Q[i] = temp[i] * (1 - temporal_filtering ) + Q[i] * temporal_filtering
#         dh[i] += cte.DT_HYDRO * Q[i]/cte.DX**2
#         h[i] += dh[i]


@ti.kernel
def graphflood_diffuse_cte_P_cte_man(z: ti.template(), h:ti.template(), Q: ti.template(), temp: ti.template(), dh: ti.template(), srecs: ti.template(), LM: ti.template(), temporal_filtering:cte.FLOAT_TYPE_TI):
    """
    NEXT STEPS::add a tag that propagate from local minimas and reroute from corrected receivers

    Author: B.G.
    """

    # Initialize precipitation input and handle boundary conditions
    for i in Q:
        temp[i] = cte.PREC * cte.DX * cte.DX  # Add precipitation as volume input
        dh[i] = 0.

    # DOES NOT WORK IN TAICHI -> outter loop always the main parallel one
    # ti.loop_config(serialize=True)
    # for damier in range(2):

    # Diffuse discharge based on slope gradients
    for i in z:

        # Skip boundary cells
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue

            # Calculate total slope gradient sum for normalization
        sums = 0.0
        msx  = 0.0
        msy  = 0.0
        mz   = (z[i]+h[i])
        tlm = True

        for k in range(4):  # Check all 4 neighbors
            j = flow.neighbourer_flat.neighbour(i, k)
            if(j == -1 or flow.neighbourer_flat.nodata(j)):
                continue
            
            mz  = ti.max(mz, (z[j]+h[j]))

            if z[j]+h[j]<z[i]+h[i]:
                tlm = False
                break

        # Skip cells with no downslope neighbors
        if tlm:
            LM[i] = True
            dh[i] += 5e-3 # mz - z[i] + 5e-3
            # continue

        for k in range(4):  # Check all 4 neighbors
            j = flow.neighbourer_flat.neighbour(i, k)
            if(j == -1 or flow.neighbourer_flat.nodata(j)):
                continue

            if ( LM[i] and srecs[j] == i):
                continue

            ts = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
            sums += ts
            msx = ti.max(ts, msx) if k == 1 or k == 2 else msx
            msy = ti.max(ts, msy) if k == 0 or k == 3 else msy

        
   
        if LM[i] and LM[srecs[i]] == False:
            LM[srecs[i]] = True

        if(LM[i] and sums == 0.):
            ti.atomic_add(temp[srecs[i]], Q[i])
        else:
            # Distribute discharge proportionally to slope gradients
            for k in range(4):
                j = flow.neighbourer_flat.neighbour(i, k)
                if(j == -1 or flow.neighbourer_flat.nodata(j)):
                    continue
                if ( LM[i] and srecs[j] == i):
                    continue
                tS = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
                ti.atomic_add(temp[j], tS / sums * Q[i])  # Add proportional discharge

        norms = ti.math.sqrt(msx**2 + msy**2) # if (sums > 0.) else ti.max((z[i]+h[i] - z[srecs[i]] - h[srecs[i]])/cte.DX,1e-3)
        norms = ti.max(norms, 1e-4)
        Qo = cte.DX * h[i] ** (5.0 / 3.0) / cte.MANNING * ti.math.sqrt(norms)
        dh[i] -= cte.DT_HYDRO * Qo/cte.DX**2



    # Update discharge field with diffused values
    for i in Q:
        Q[i] = temp[i] * (1 - temporal_filtering ) + Q[i] * temporal_filtering
        dh[i] += cte.DT_HYDRO * Q[i]/cte.DX**2
        h[i] += dh[i]
        if(h[i] < 0):
            h[i] = 0
        #     print("JLKFSDFSDJLK")



@ti.kernel
def graphflood_diffuse_cte_P_cte_man_dt(z: ti.template(), h:ti.template(), Q: ti.template(), temp: ti.template(), 
    dh: ti.template(), srecs: ti.template(), temporal_filtering:ti.f32, dt_local :cte.FLOAT_TYPE_TI):
    """
    NEXT STEPS::add a tag that propagate from local minimas and reroute from corrected receivers

    Author: B.G.
    """

    # Initialize precipitation input and handle boundary conditions
    for i in Q:
        temp[i] = cte.PREC * cte.DX * cte.DX  # Add precipitation as volume input
        dh[i] = 0.

    # DOES NOT WORK IN TAICHI -> outter loop always the main parallel one
    # ti.loop_config(serialize=True)
    # for damier in range(2):

    # Diffuse discharge based on slope gradients
    for i in z:

        # Skip boundary cells
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue

            # Calculate total slope gradient sum for normalization
        sums = 0.0
        msx  = 0.0
        msy  = 0.0
        mz   = (z[i]+h[i])

        for k in range(4):  # Check all 4 neighbors
            j = flow.neighbourer_flat.neighbour(i, k)
            if(j == -1 or flow.neighbourer_flat.nodata(j)):
                continue

            ts = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
            sums += ts
            msx = ti.max(ts, msx) if k == 1 or k == 2 else msx
            msy = ti.max(ts, msy) if k == 0 or k == 3 else msy

        


        if(sums == 0.):
            ti.atomic_add(temp[srecs[i]], Q[i])
            dh[i] += 5e-3
        else:
            # Distribute discharge proportionally to slope gradients
            for k in range(4):
                j = flow.neighbourer_flat.neighbour(i, k)

                if(j == -1 or flow.neighbourer_flat.nodata(j)):
                    continue

                tS = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
                ti.atomic_add(temp[j], tS / sums * Q[i])  # Add proportional discharge

        norms = ti.math.sqrt(msx**2 + msy**2) # if (sums > 0.) else ti.max((z[i]+h[i] - z[srecs[i]] - h[srecs[i]])/cte.DX,1e-3)
        norms = ti.max(norms, 1e-6)
        Qo = cte.DX * h[i] ** (5.0 / 3.0) / cte.MANNING * ti.math.sqrt(norms)
        dh[i] -= dt_local * Qo/cte.DX**2


    # Update discharge field with diffused values
    for i in Q:
        Q[i] = (1 - temporal_filtering ) * Q[i] + temp[i] * temporal_filtering
        dh[i] += dt_local * Q[i]/cte.DX**2
        h[i] += dh[i]
        if(h[i] < 0):
            h[i] = 0

@ti.kernel
def _legacy_graphflood_diffuse_cte_P_cte_man_dt(z: ti.template(), h:ti.template(), Q: ti.template(), temp: ti.template(), 
    dh: ti.template(), srecs: ti.template(), LM: ti.template(), temporal_filtering:ti.f32, dt_local :cte.FLOAT_TYPE_TI):
    """
    NEXT STEPS::add a tag that propagate from local minimas and reroute from corrected receivers

    Author: B.G.
    """

    # Initialize precipitation input and handle boundary conditions
    for i in Q:
        temp[i] = cte.PREC * cte.DX * cte.DX  # Add precipitation as volume input
        dh[i] = 0.

    # DOES NOT WORK IN TAICHI -> outter loop always the main parallel one
    # ti.loop_config(serialize=True)
    # for damier in range(2):

    # Diffuse discharge based on slope gradients
    for i in z:


        # Skip boundary cells
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue

            # Calculate total slope gradient sum for normalization
            sums = 0.0
            msx  = 0.0
            msy  = 0.0
            mz   = (z[i]+h[i])
            tlm = True

            for k in range(4):  # Check all 4 neighbors
                j = flow.neighbourer_flat.neighbour(i, k)
                if(j == -1 or flow.neighbourer_flat.nodata(j)):
                    continue
                
                mz  = ti.max(mz, (z[j]+h[j]))

                if z[j]+h[j]<z[i]+h[i]:
                    tlm = False
                    break

            # Skip cells with no downslope neighbors
            if tlm:
                LM[i] = True
                dh[i] += 5e-3 # mz - z[i] + 5e-3
                # continue

            for k in range(4):  # Check all 4 neighbors
                j = flow.neighbourer_flat.neighbour(i, k)
                if(j == -1 or flow.neighbourer_flat.nodata(j)):
                    continue

                if ( LM[i] and srecs[j] == i):
                    continue

                ts = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
                sums += ts
                msx = ti.max(ts, msx) if k == 1 or k == 2 else msx
                msy = ti.max(ts, msy) if k == 0 or k == 3 else msy

        
   
        if LM[i] and LM[srecs[i]] == False:
            LM[srecs[i]] = True

        if(LM[i] and sums == 0.):
            ti.atomic_add(temp[srecs[i]], Q[i])
        else:
            # Distribute discharge proportionally to slope gradients
            for k in range(4):
                j = flow.neighbourer_flat.neighbour(i, k)
                if(j == -1 or flow.neighbourer_flat.nodata(j)):
                    continue
                if ( LM[i] and srecs[j] == i):
                    continue
                tS = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
                ti.atomic_add(temp[j], tS / sums * Q[i])  # Add proportional discharge

        norms = ti.math.sqrt(msx**2 + msy**2) # if (sums > 0.) else ti.max((z[i]+h[i] - z[srecs[i]] - h[srecs[i]])/cte.DX,1e-3)
        norms = ti.max(norms, 1e-4)
        Qo = cte.DX * h[i] ** (5.0 / 3.0) / cte.MANNING * ti.math.sqrt(norms)
        dh[i] -= dt_local * Qo/cte.DX**2


    # Update discharge field with diffused values
    for i in Q:
        Q[i] = temp[i] * (1 - temporal_filtering ) + Q[i] * temporal_filtering
        dh[i] += dt_local * Q[i]/cte.DX**2
        h[i] += dh[i]
        if(h[i] < 0):
            h[i] = 0
        #     print("JLKFSDFSDJLK")

@ti.kernel
def graphflood_cte_man_dt_nopropag(z: ti.template(), h:ti.template(), Q: ti.template(),
    dh: ti.template(), dt_local :cte.FLOAT_TYPE_TI):
    """
    NEXT STEPS::add a tag that propagate from local minimas and reroute from corrected receivers

    Author: B.G.
    """

    # Initialize precipitation input and handle boundary conditions
    for i in Q:
        dh[i] = 0.

    # DOES NOT WORK IN TAICHI -> outter loop always the main parallel one
    # ti.loop_config(serialize=True)
    # for damier in range(2):

    # Diffuse discharge based on slope gradients
    for i in z:

        # Skip boundary cells
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue

        # Calculate total slope gradient sum for normalization
        msx  = 0.0
        msy  = 0.0
        for k in range(4):  # Check all 4 neighbors
            j = flow.neighbourer_flat.neighbour(i, k)
            if(j == -1 or flow.neighbourer_flat.nodata(j)):
                continue

            ts = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
            msx = ti.max(ts, msx) if k == 1 or k == 2 else msx
            msy = ti.max(ts, msy) if k == 0 or k == 3 else msy

        norms = ti.math.sqrt(msx**2 + msy**2) # if (sums > 0.) else ti.max((z[i]+h[i] - z[srecs[i]] - h[srecs[i]])/cte.DX,1e-3)
        # norms = ti.max(norms, 1e-4)
        # if msx+msy == 0.:
        #     dh[i] += ti.random() * 1e-2
            
        Qo = cte.DX * h[i] ** (5.0 / 3.0) / cte.MANNING * ti.math.sqrt(norms)
        dh[i] -= dt_local * Qo/cte.DX**2


    # Update discharge field with diffused values
    for i in Q:

        # Skip boundary cells
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            h[i] = 0
            continue

        dh[i] += dt_local * Q[i]/cte.DX**2
        h[i] += dh[i]
        if(h[i] < 0):
            h[i] = 0


@ti.kernel
def graphflood_dt_splat(h:ti.template(), Q: ti.template(), dt_local :cte.FLOAT_TYPE_TI):
    """
    NEXT STEPS::add a tag that propagate from local minimas and reroute from corrected receivers

    Author: B.G.
    """

    coeff = dt_local/(cte.DX**2)

    # Diffuse discharge based on slope gradients
    for i in h:

        # Skip boundary cells
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue

        h[i] += Q[i] * coeff


@ti.kernel
def graphflood_cte_man_dt_nopropag_mask(z: ti.template(), h:ti.template(), Q: ti.template(),
    dh: ti.template(), mask:ti.template(), dt_local :cte.FLOAT_TYPE_TI):
    """
    NEXT STEPS::add a tag that propagate from local minimas and reroute from corrected receivers

    Author: B.G.
    """

    # Initialize precipitation input and handle boundary conditions
    for i in Q:
        dh[i] = 0.

    # DOES NOT WORK IN TAICHI -> outter loop always the main parallel one
    # ti.loop_config(serialize=True)
    # for damier in range(2):

    # Diffuse discharge based on slope gradients
    for i in z:

        if mask[i] == False:
            continue

        # Skip boundary cells
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue

        # Calculate total slope gradient sum for normalization
        msx  = 0.0
        msy  = 0.0
        for k in range(4):  # Check all 4 neighbors
            j = flow.neighbourer_flat.neighbour(i, k)
            if(j == -1 or flow.neighbourer_flat.nodata(j)):
                continue

            ts = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
            msx = ti.max(ts, msx) if k == 1 or k == 2 else msx
            msy = ti.max(ts, msy) if k == 0 or k == 3 else msy


        norms = ti.math.sqrt(msx**2 + msy**2) # if (sums > 0.) else ti.max((z[i]+h[i] - z[srecs[i]] - h[srecs[i]])/cte.DX,1e-3)
        # norms = ti.max(norms, 1e-4)
        # if msx+msy == 0.:
        #     dh[i] += ti.random() * 1e-2
            
        Qo = cte.DX * h[i] ** (5.0 / 3.0) / cte.MANNING * ti.math.sqrt(norms)
        dh[i] -= dt_local * Qo/cte.DX**2


    # Update discharge field with diffused values
    for i in Q:
        if mask[i] == False:
            continue

        # Skip boundary cells
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            h[i] = 0
            continue

        dh[i] += dt_local * Q[i]/cte.DX**2
        h[i] += dh[i]
        if(h[i] < 0):
            h[i] = 0

@ti.kernel
def graphflood_cte_man_analytical(z: ti.template(), h:ti.template(), Q: ti.template(), temporal_filtering:cte.FLOAT_TYPE_TI, Q_filtering:cte.FLOAT_TYPE_TI):
    """
    NEXT STEPS::add a tag that propagate from local minimas and reroute from corrected receivers

    Author: B.G.
    """

    # Diffuse discharge based on slope gradients
    for i in z:

        # Skip boundary cells
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue

        # Calculate total slope gradient sum for normalization
        msx  = 0.0
        msy  = 0.0
        for k in range(4):  # Check all 4 neighbors
            j = flow.neighbourer_flat.neighbour(i, k)
            if(j == -1 or flow.neighbourer_flat.nodata(j)):
                continue

            ts = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
            msx = ti.max(ts, msx) if k == 1 or k == 2 else msx
            msy = ti.max(ts, msy) if k == 0 or k == 3 else msy

        if(msx == 0 and msy == 0):
            msx = 1e-2
            msy = 1e-2

        norms = ti.math.sqrt(msx**2 + msy**2)
        tQ = Q[i]  
        if(Q_filtering > 0):

            Qo = cte.DX * h[i] ** (5.0 / 3.0) / cte.MANNING * ti.math.sqrt(norms)
            tQ = (1- Q_filtering) * tQ + Q_filtering * Qo

        h[i] = (1- temporal_filtering) * h[i] + temporal_filtering * (tQ * cte.MANNING / ti.math.sqrt(norms) / cte.DX)**(3./5.)

@ti.kernel
def graphflood_cte_man_analytical_mask(z: ti.template(), h:ti.template(), Q: ti.template(), mask:ti.template(), temporal_filtering:cte.FLOAT_TYPE_TI):
    """

    Author: B.G.
    """

    # Diffuse discharge based on slope gradients
    for i in z:
        if mask[i] == False:
            continue

        # Skip boundary cells
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue

        # Calculate total slope gradient sum for normalization
        msx  = 0.0
        msy  = 0.0
        for k in range(4):  # Check all 4 neighbors
            j = flow.neighbourer_flat.neighbour(i, k)
            if(j == -1 or flow.neighbourer_flat.nodata(j)):
                continue

            ts = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
            msx = ti.max(ts, msx) if k == 1 or k == 2 else msx
            msy = ti.max(ts, msy) if k == 0 or k == 3 else msy

        if(msx == 0 and msy == 0):
            for k in range(4):  # Check all 4 neighbors
                j = flow.neighbourer_flat.neighbour(i, k)
                if(j == -1 or flow.neighbourer_flat.nodata(j)):
                    continue

                ts = ti.max(0.0, ((z[i] - z[j]) / cte.DX) if j != -1 else 0.0)
                msx = ti.max(ts, msx) if k == 1 or k == 2 else msx
                msy = ti.max(ts, msy) if k == 0 or k == 3 else msy

        if(msx == 0 and msy == 0):
            msx = 1e-3
            msy = 1e-3

        norms = ti.math.sqrt(msx**2 + msy**2)
            
        # Qo = cte.DX * h[i] ** (5.0 / 3.0) / cte.MANNING * ti.math.sqrt(norms)
        h[i] = (1- temporal_filtering) * h[i] + temporal_filtering * (Q[i] * cte.MANNING / ti.math.sqrt(norms) / cte.DX)**(3./5.)

@ti.kernel
def graphflood_get_Qo(z: ti.template(), h:ti.template(), Qo: ti.template()):
    """

    Author: B.G.
    """

    # Diffuse discharge based on slope gradients
    for i in z:
        # Skip boundary cells
        if flow.neighbourer_flat.can_leave_domain(i):
            continue

        # Calculate total slope gradient sum for normalization
        sums = 0.0
        msx  = 0.0
        msy  = 0.0
    
        for k in ti.static(range(4)):  # Check all 4 neighbors
            j = flow.neighbourer_flat.neighbour(i, k)
            ts = ti.max(0.0, (((z[i]+h[i]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
            sums += ts
            msx = ti.max(ts, msx) if k == 1 or k == 2 else msx
            msy = ti.max(ts, msy) if k == 0 or k == 3 else msy


        # Skip cells with no downslope neighbors
        if sums == 0.0:
            Qo[i] = 0.
            continue

        norms = ti.math.sqrt(msx**2 + msy**2)
        Qo[i] = cte.DX * h[i] ** (5.0 / 3.0) / cte.MANNING * ti.math.sqrt(norms)
        

@ti.func
def nbor(i, k: ti.template()):
    return flow.neighbourer_flat.neighbour(i, k)

@ti.func
def val_at(f: ti.template(), idx: ti.i32, self_idx: ti.i32):
    return f[self_idx] if idx < 0 else f[idx]

@ti.func
def signf(x):
    return ti.cast(1.0, cte.FLOAT_TYPE_TI) if x >= 0 else ti.cast(-1.0, cte.FLOAT_TYPE_TI)

# # ----------------- velocity from -∇(z + 1/2 h^2) -----------------
# @ti.kernel
# def compute_velocity_flat(z: ti.template(), h: ti.template(),
#                           ux: ti.template(), uy: ti.template(),
#                           vscale: ti.f32):
#     inv2dx = 0.5 / cte.DX  # dx = dy = cte.DX
#     inv2dy = 0.5 / cte.DX
#     nman = ti.max(cte.MANNING, 1e-6)

#     for i in z:
#         # neighbors
#         iT = nbor(i, 0)  # top
#         iL = nbor(i, 1)  # left
#         iR = nbor(i, 2)  # right
#         iB = nbor(i, 3)  # bottom

#         ux[i] = vscale * 

# # ----------------- conservative update of h^(5/3) -----------------
# @ti.kernel
# def diffuse_step_flat(h: ti.template(), ux: ti.template(), uy: ti.template(),
#                       S: ti.template(), k: ti.template(), kn: ti.template(),
#                       dt: ti.f32):
#     dx = cte.DX
#     dy = cte.DX

#     # k = h^(5/3)
#     for i in h:
#         k[i] = ti.pow(ti.max(0.0,h[i]), 5.0 / 3.0)

#     # flux divergence on faces using neighbor averages
#     for i in h:
#         iT = nbor(i, 0)
#         iL = nbor(i, 1)
#         iR = nbor(i, 2)
#         iB = nbor(i, 3)

#         # face-averaged k
#         kR = 0.5 * (k[i] + val_at(k, iR, i))
#         kL = 0.5 * (k[i] + val_at(k, iL, i))
#         kB = 0.5 * (k[i] + val_at(k, iB, i))
#         kT = 0.5 * (k[i] + val_at(k, iT, i))

#         # face-averaged velocities
#         uXR = 0.5 * (ux[i] + val_at(ux, iR, i))
#         uXL = 0.5 * (ux[i] + val_at(ux, iL, i))
#         uYB = 0.5 * (uy[i] + val_at(uy, iB, i))
#         uYT = 0.5 * (uy[i] + val_at(uy, iT, i))

#         # fluxes
#         FxR = kR * uXR
#         FxL = kL * uXL
#         FyB = kB * uYB
#         FyT = kT * uYT

#         div = (FxR - FxL) / dx + (FyB - FyT) / dy
#         source = ti.max(0.0, S[i])

#         kn[i] = k[i] + dt * (div + source)

#     # back to h, enforce non-negativity
#     for i in h:
#         h[i] = ti.pow(ti.max(0.0, kn[i]), 3.0 / 5.0)



@ti.kernel
def run_vdb23_step5(z:ti.template(), h:ti.template(), dh:ti.template(), S:ti.template(), ux:ti.template(), uy:ti.template(), dt:ti.f32, omega:cte.FLOAT_TYPE_TI):

    for i in z:

        if flow.neighbourer_flat.nodata(i):
            continue 

        for tk in range(2):
            k = tk + 2
            j = flow.neighbourer_flat.neighbour(i,k)
            if(j == -1 or flow.neighbourer_flat.nodata(j)):
                continue
            if(h[i] == 0 and h[j] == 0):
                continue
            
            # topo gradient
            dzdx = z[i]-z[j]
            dzdx /= cte.DX
            
            # pressure head
            dhdx = h[i]**2 - h[j]**2
            dhdx /= 2*cte.DX

            # Upwind
            # th = ti.max(h[i],h[j])
            th = ti.math.max(z[i] + h[i], z[j] + h[j]) - ti.math.max(
                    z[j], z[i]
                )
            th = ti.max(th,0.)

            if tk == 0:
                tu = th**(1.5)/cte.MANNING * (dzdx + dhdx)
                ux[i] = (1-omega) * ux[i] + omega * ti.math.sign(tu)*ti.sqrt(ti.abs(tu) )
            else:
                tu = th**(1.5)/cte.MANNING * (dzdx + dhdx)
                uy[i] = (1-omega) * uy[i] + omega * ti.math.sign(tu)*ti.sqrt(ti.abs(tu) )

    for i in h:
        h[i] = (h[i]**(5./3.)) if h[i]> 0 else 0

    for i in h:

        if flow.neighbourer_flat.nodata(i):
            continue 

        divu = 0.

        maxu = 0.

        for tk in range(2):
            k = tk
            j = flow.neighbourer_flat.neighbour(i,k)
            if(j == -1 or flow.neighbourer_flat.nodata(j)):
                continue
            if(h[i] == 0 and h[j] == 0):
                continue

            tu = (ux[j] if tk==1 else uy[j])
            th = h[i] if tu<0 else h[j]
            # th = ti.math.max(z[i] + h[i], z[j] + h[j]) - ti.math.max(
            #         z[j], z[i]
            #     )
            th = ti.max(0., th)
            divu +=  tu*th
            maxu = ti.max(maxu, ti.abs(tu))

        for tk in range(2):
            j = flow.neighbourer_flat.neighbour(i,tk+2)
            if(j == -1 or flow.neighbourer_flat.nodata(j)):
                continue
            if(h[i] == 0 and h[j] == 0):
                continue

            tu = (ux[i] if tk==0 else uy[i])
            th = h[i] if tu>0 else h[j]
            # th = ti.math.max(z[i] + h[i], z[j] + h[j]) - ti.math.max(
            #         z[j], z[i]
                # )
            th = ti.max(0., th)
            divu -=  tu*th
            maxu = ti.max(maxu, ti.abs(tu))

        # dt = cfl * cte.DX/(maxu + 1e-3)
        dh[i] = dt * (divu + S[i])

    for i in h:
        h[i] += dh[i]
        
        if(h[i]<0):
            h[i] = 0.

        h[i] = h[i]**(3./5.)


@ti.kernel
def run_BG24_dyn(z:ti.template(), h:ti.template(), dh:ti.template(), S:ti.template(), qx:ti.template(), qy:ti.template(), dt:ti.f32, omega:cte.FLOAT_TYPE_TI):

    for i in z:

        if flow.neighbourer_flat.nodata(i):
            continue 

        for tk in range(2):
            k = tk + 2
            j = flow.neighbourer_flat.neighbour(i,k)
            if(j == -1 or flow.neighbourer_flat.nodata(j)):
                continue
            if(h[i] == 0 and h[j] == 0):
                continue
            
            # topo gradient
            dzdx = (z[i]+h[i])-(z[j]+h[j])
            dzdx /= cte.DX
            

            # Upwind
            th = ti.max(h[i],h[j])
            # th = ti.math.max(z[i] + h[i], z[j] + h[j]) - ti.math.max(
            #         z[j], z[i]
            #     )
            th = ti.max(th,0.)

            if tk == 0:
                tq = th**(5./3.)/cte.MANNING * ti.math.sqrt(ti.abs(dzdx))
                qx[i] = ti.math.sign(dzdx)*ti.abs(tq)
            else:
                tq = th**(5./3.)/cte.MANNING * ti.math.sqrt(ti.abs(dzdx))
                qy[i] = ti.math.sign(dzdx)*ti.abs(tq)


    for i in h:

        if flow.neighbourer_flat.nodata(i):
            continue 

        divu = 0.

        for tk in range(2):
            k = tk
            j = flow.neighbourer_flat.neighbour(i,k)
            if(j == -1 or flow.neighbourer_flat.nodata(j)):
                continue
            if(h[i] == 0 and h[j] == 0):
                continue

            tq = (qx[j] if tk==1 else qy[j])
            divu +=  tq

        for tk in range(2):
            j = flow.neighbourer_flat.neighbour(i,tk+2)
            if(j == -1 or flow.neighbourer_flat.nodata(j)):
                continue
            if(h[i] == 0 and h[j] == 0):
                continue

            tq = (qx[i] if tk==0 else qy[i])
            divu -=  tq

        dh[i] = dt * (divu/cte.DX + S[i])

    for i in h:
        h[i] = (1-omega) * h[i] + omega * (h[i] + dh[i])
        
        if(h[i]<0):
            h[i] = 0.


@ti.kernel
def run_BG24_add(h:ti.template(), Qin:ti.template(), dt:ti.f32, omega:cte.FLOAT_TYPE_TI):

    for i in h:

        if flow.neighbourer_flat.nodata(i):
            continue 

        h[i] = (1-omega) * h[i] + omega* dt * Qin[i]/(cte.DX**2)
        







        













@ti.kernel
def pgraph_v0(particles:ti.template(), z:ti.template(), h:ti.template(), dh:ti.template(), S:ti.template(), Q:ti.template(), Q_:ti.template(), manning:ti.f32, dt:ti.f32, omega:ti.f32):


    for pi in particles:
        i = particles[pi].pos
        Q_[i] = S[i] * cte.DX**2
        dh[i] = 0.

    for pi in particles:

        i = particles[pi].pos

        
        msx  = 0.0
        msy  = 0.0

        for tk in range(4):  # Check all 4 neighbors
            tj = flow.neighbourer_flat.neighbour(i, tk)
            if(tj == -1 or flow.neighbourer_flat.nodata(tj)):
                continue

            if(z[tj]+h[tj]>z[i]+h[i]):
                sums = 0.0
                tss = 0.0
                for k in range(4):  # Check all 4 neighbors

                    j = flow.neighbourer_flat.neighbour(tj, k)
                    
                    if(j == -1 or flow.neighbourer_flat.nodata(j)):
                        continue

                    ts = ti.max(0.0, (((z[tj]+h[tj]) - (z[j]+h[j])) / cte.DX) if j != -1 else 0.0)
                    if j == i:
                        tss = ts
                    sums += ts
                if tss>0.:
                    Q_[i] += Q[tj]*tss/sums
            else:
                ts = ti.max(0.0, (((z[i]+h[i]) - (z[tj]+h[tj])) / cte.DX) if tj != -1 else 0.0)
                msx = ti.max(ts, msx) if tk == 1 or tk == 2 else msx
                msy = ti.max(ts, msy) if tk == 0 or tk == 3 else msy


        if(msx >0 and msy > 0 and h[i]>0):
            norm = ti.math.sqrt(msx**2 + msy**2)
            norm = ti.max(norm,1e-4)
            tQo = cte.DX/manning * h[i]**(5./3.)*ti.math.sqrt(norm)
            dh[i] -= dt*tQo




    for pi in particles:
        i = particles[pi].pos

        Q[i] = Q_[i]
        dh[i] += Q[i] * dt
        dh[i] /= (cte.DX**2)
        h[i] += dh[i]

        if(h[i]<0):
            h[i]=0


@ti.func
def update_wxy(index, wx, wy, z, h):
    '''
    this function updates wx and wy for a given node. Loop thorugh neighbours' z and h, calculate the sum of the slope in z+h and weights are locals/sums for node i
    '''
    sums = 0.0
    process = ti.Vector([False,False,False,False])
    slopes = ti.Vector([0.,0.,0.,0.])
    neigh = ti.Vector(
        [flow.neighbourer_flat.neighbour(index, 0),
         flow.neighbourer_flat.neighbour(index, 1),
         flow.neighbourer_flat.neighbour(index, 2),
         flow.neighbourer_flat.neighbour(index, 3)]
    )


    for k in range(4):
        j = neigh[k]
        if j == -1 or flow.neighbourer_flat.nodata(j):
            continue

        slope = ((z[index] + h[index]) - (z[j] + h[j])) / cte.DX

        process[k] = True if slope > 0 else False
        slopes[k] = slope

        # Only sum positive slopes (downhill/outflow directions)
        if slope > 0:
            sums += slope

    if sums > 0:
        if(process[0]):
            wy[neigh[0]] = -slopes[0] / sums
        if(process[1]):
            wx[neigh[1]] = -slopes[1] / sums
        if(process[2]):
            wx[index] = slopes[2] / sums
        if(process[3]):
            wy[index] = slopes[3] / sums
    else:
        h[index] += 1e-2*ti.random()





@ti.func
def update_Q(index, Q, Q_, wx, wy):
    '''
    Accumulate Q contributions from all uphill neighbors.

    Convention: wx[i] is weight on link from i to right, wy[i] is weight on link from i to bottom
    Positive weight = flow in positive direction (right/down)
    Negative weight = flow in negative direction (left/up)
    Only uphill nodes set weights, so we collect inflows from all 4 neighbors
    '''
    # From left: if wx[left] > 0, left distributed flow rightward to index
    j_left = flow.neighbourer_flat.neighbour(index, 1)
    if j_left != -1 and not flow.neighbourer_flat.nodata(j_left):
        if wx[j_left] > 0:
            Q_[index] += wx[j_left] * Q[j_left]

    # From top: if wy[top] > 0, top distributed flow downward to index
    j_top = flow.neighbourer_flat.neighbour(index, 0)
    if j_top != -1 and not flow.neighbourer_flat.nodata(j_top):
        if wy[j_top] > 0:
            Q_[index] += wy[j_top] * Q[j_top]

    # From right: if wx[index] < 0, right distributed flow leftward to index
    j_right = flow.neighbourer_flat.neighbour(index, 2)
    if j_right != -1 and not flow.neighbourer_flat.nodata(j_right):
        if wx[index] < 0:
            Q_[index] += (-wx[index]) * Q[j_right]

    # From bottom: if wy[index] < 0, bottom distributed flow upward to index
    j_bottom = flow.neighbourer_flat.neighbour(index, 3)
    if j_bottom != -1 and not flow.neighbourer_flat.nodata(j_bottom):
        if wy[index] < 0:
            Q_[index] += (-wy[index]) * Q[j_bottom]


@ti.func
def renormalize_outflows(index, wx, wy):
    '''
    Renormalize outflow weights at node index to ensure sum of |outflows| = 1.
    Does not recalculate based on slopes, just ensures artificial mass balance.

    For a node, outflows are:
    - wx[index] if positive (to right)
    - wy[index] if positive (to bottom)
    - wx[left] if negative (to left, stored on left's link)
    - wy[top] if negative (to top, stored on top's link)
    '''
    j_left = flow.neighbourer_flat.neighbour(index, 1)
    j_top = flow.neighbourer_flat.neighbour(index, 0)

    # Sum absolute values of all outflows
    sum_abs = 0.0

    # Check right outflow
    if wx[index] > 0:
        sum_abs += wx[index]

    # Check bottom outflow
    if wy[index] > 0:
        sum_abs += wy[index]

    # Check left outflow
    if j_left != -1 and not flow.neighbourer_flat.nodata(j_left):
        if wx[j_left] < 0:
            sum_abs += -wx[j_left]

    # Check top outflow
    if j_top != -1 and not flow.neighbourer_flat.nodata(j_top):
        if wy[j_top] < 0:
            sum_abs += -wy[j_top]

    # Renormalize if there are outflows
    if sum_abs > 0:
        if wx[index] > 0:
            wx[index] = wx[index] / sum_abs

        if wy[index] > 0:
            wy[index] = wy[index] / sum_abs

        if j_left != -1 and not flow.neighbourer_flat.nodata(j_left):
            if wx[j_left] < 0:
                wx[j_left] = wx[j_left] / sum_abs  # Keep negative sign

        if j_top != -1 and not flow.neighbourer_flat.nodata(j_top):
            if wy[j_top] < 0:
                wy[j_top] = wy[j_top] / sum_abs  # Keep negative sign


@ti.kernel
def compute_weights_wxy(wx: ti.template(), wy: ti.template(), z: ti.template(), h: ti.template()):
    """
    Compute flow weights for all nodes based on z+h surface slopes.

    Updates wx (right) and wy (bottom) weights for each node by computing
    slope-weighted distributions toward lower neighbors.

    Args:
        wx: Weight field for rightward flow
        wy: Weight field for downward flow
        z: Elevation field
        h: Water depth field
    """
    for i in z:
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue
        update_wxy(i, wx, wy, z, h)


@ti.kernel
def diffuse_Q_with_weights(Q: ti.template(), Q_: ti.template(), wx: ti.template(), wy: ti.template(), S: ti.template(), omega:ti.f32):
    """
    Diffuse discharge Q using pre-computed weights wx, wy.

    Three-step process:
    1. Initialize Q_ with source term S
    2. Accumulate weighted contributions from neighbors
    3. Copy result back to Q

    Args:
        Q: Discharge field (input/output)
        Q_: Temporary discharge field
        wx: Weight field for rightward flow
        wy: Weight field for downward flow
        S: Source term field
    """
    # First loop: init Q_ from source
    for i in Q:
        Q_[i] = S[i]

    # Second loop: accumulate weighted Q contributions
    for i in Q:
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue
        update_Q(i, Q, Q_, wx, wy)

    # Third loop: copy back
    for i in Q:
        Q[i] = (1-omega) * Q[i] + omega * Q_[i]



@ti.kernel
def pgraph_Q(particles: ti.template(), wx: ti.template(), wy: ti.template(), z: ti.template(),
             h: ti.template(), Q: ti.template(), Q_: ti.template(), S: ti.template(), mask_Q: ti.template()):
    """
    Sparse discharge diffusion for particle-occupied nodes and their neighbors.

    Args:
        particles: Particle field with .pos attribute
        wx: Weight field for rightward flow
        wy: Weight field for downward flow
        z: Elevation field
        h: Water depth field
        Q: Discharge field (input/output)
        Q_: Temporary discharge field
        S: Source term field
        mask_Q: ti.u8 mask field for tracking processed nodes
    """
    # Reset mask to 0
    for i in mask_Q:
        mask_Q[i] = ti.u8(0)

    # For all particle positions: update weights and flag mask
    for pi in particles:
        i = particles[pi].pos
        update_wxy(i, wx, wy, z, h)
        mask_Q[i] = ti.u8(1)

    # For all particle positions: process neighbors
    for pi in particles:
        i = particles[pi].pos
        for k in range(4):
            j = flow.neighbourer_flat.neighbour(i, k)
            if j == -1 or flow.neighbourer_flat.nodata(j):
                continue

            # Atomically check and flag neighbor
            old_val = ti.atomic_max(mask_Q[j], ti.u8(1))
            if old_val == ti.u8(0):
                # First thread to reach this neighbor - renormalize it
                renormalize_outflows(j, wx, wy)

    # Initialize Q_ from S for active nodes (mask=1 means needs processing)
    for pi in particles:
        i = particles[pi].pos
        Q_[i] = S[i]

        for k in range(4):
            j = flow.neighbourer_flat.neighbour(i, k)
            if j != -1 and not flow.neighbourer_flat.nodata(j):
                if mask_Q[j] == ti.u8(1):
                    Q_[j] = S[j]

    # Accumulate Q contributions: process particle nodes and atomically check neighbors
    for pi in particles:
        i = particles[pi].pos
        update_Q(i, Q, Q_, wx, wy)
        mask_Q[i] = ti.u8(2)

        for k in range(4):
            j = flow.neighbourer_flat.neighbour(i, k)
            if j == -1 or flow.neighbourer_flat.nodata(j):
                continue

            # Atomically check and flag neighbor
            old_val = ti.atomic_max(mask_Q[j], ti.u8(2))
            if old_val == ti.u8(1):
                update_Q(j, Q, Q_, wx, wy)

    # Copy back for processed nodes (mask=2)
    for pi in particles:
        i = particles[pi].pos
        Q[i] = Q_[i]

        for k in range(4):
            j = flow.neighbourer_flat.neighbour(i, k)
            if j != -1 and not flow.neighbourer_flat.nodata(j):
                if mask_Q[j] == ti.u8(2):
                    Q[j] = Q_[j]









@ti.func
def _compute_sw(index, z, h):
    '''
    this function updates wx and wy for a given node. Loop thorugh neighbours' z and h, calculate the sum of the slope in z+h and weights are locals/sums for node i
    '''
    msx = 0.0
    msy = 0.0
    process = ti.Vector([False,False,False,False])
    slopes = ti.Vector([0.,0.,0.,0.])
    neigh = ti.Vector(
        [flow.neighbourer_flat.neighbour(index, 0),
         flow.neighbourer_flat.neighbour(index, 1),
         flow.neighbourer_flat.neighbour(index, 2),
         flow.neighbourer_flat.neighbour(index, 3)]
    )


    if neigh[0]>-1:
        slope = ((z[index] + h[index]) - (z[neigh[0]] + h[neigh[0]])) / cte.DX
        if slope > msy:
            msy = slope
    if neigh[3]>-1:
        slope = ((z[index] + h[index]) - (z[neigh[3]] + h[neigh[3]])) / cte.DX
        if slope > msy:
            msy = slope

    if neigh[1]>-1:
        slope = ((z[index] + h[index]) - (z[neigh[1]] + h[neigh[1]])) / cte.DX
        if slope > msx:
            msx = slope
    if neigh[2]>-1:
        slope = ((z[index] + h[index]) - (z[neigh[2]] + h[neigh[2]])) / cte.DX
        if slope > msx:
            msx = slope

    norm = ti.math.sqrt(msx**2 + msy**2)

    return norm

@ti.kernel
def compute_sw(z:ti.template(),h:ti.template(),sw:ti.template()):

    for i in z:
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            sw[i] = 0.
            continue

        sw[i] = _compute_sw(i,z,h)

@ti.func
def _compute_tau(index, z, h,rho):
    '''
    this function updates wx and wy for a given node. Loop thorugh neighbours' z and h, calculate the sum of the slope in z+h and weights are locals/sums for node i
    '''
    sw = _compute_sw(index,z,h)
    tau = sw*h[index]*rho*9.81

    return tau

@ti.kernel
def compute_tau(z:ti.template(),h:ti.template(),tau:ti.template(), rho:ti.f32):

    for i in z:
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            tau[i] = 0.
            continue

        tau[i] = _compute_tau(i,z,h, rho)

















































# ---------------- Projection-diffusion ----------------

@ti.kernel
def init_eta(eta: ti.template(), z: ti.template(), h: ti.template()):
    for i in z:
        eta[i] = z[i] + h[i]

@ti.kernel
def copy_field(dst: ti.template(), src: ti.template()):
    for i in src:
        dst[i] = src[i]

@ti.kernel
def project_floor(eta: ti.template(), z: ti.template()):
    for i in z:
        if flow.neighbourer_flat.nodata(i):
            continue
        eta[i] = ti.max(eta[i], z[i])

@ti.kernel
def compute_h_from_eta(h: ti.template(), eta: ti.template(), z: ti.template()):
    for i in z:
        if flow.neighbourer_flat.nodata(i):
            h[i] = 0.0
        else:
            h[i] = ti.max(0.0, eta[i] - z[i])

@ti.kernel
def max_abs_diff(a: ti.template(), b: ti.template(), acc_max: ti.template()):
    acc_max[None] = 0.0
    for i in a:
        if flow.neighbourer_flat.nodata(i):
            continue
        acc_max[None] = ti.max(acc_max[None], ti.abs(a[i] - b[i]))

@ti.kernel
def volume_shifted(eta: ti.template(), z: ti.template(),
                   c: ti.f32, acc_vol: ti.template(), cell_area: ti.f64):
    acc_vol[None] = 0.0
    for i in z:
        if flow.neighbourer_flat.nodata(i):
            continue
        acc_vol[None] += cell_area * ti.max(0.0, eta[i] + c - z[i])

@ti.kernel
def apply_shift_and_project(eta: ti.template(), z: ti.template(), c: ti.f32):
    for i in z:
        if flow.neighbourer_flat.nodata(i):
            continue
        eta[i] = ti.max(z[i], eta[i] + c)

@ti.kernel
def diffuse_rb_step(eta: ti.template(), z: ti.template(),
                    parity: ti.i32, tau: ti.f32):
    for i in z:
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue
        iy, ix = flow.neighbourer_flat.rc_from_i(i)
        if ((ix + iy) & 1) != parity:
            continue
        lap = 0.0
        for k in range(4):
            j = flow.neighbourer_flat.neighbour(i, k)
            if j == -1 or flow.neighbourer_flat.nodata(j):
                continue
            lap += (eta[j] - eta[i])
        eta[i] = eta[i] + tau * lap  # explicit on this color

@ti.kernel
def diffuse_rb_step_aniso(eta: ti.template(), z: ti.template(),
                          parity: ti.i32, tau: ti.f32, sigma: ti.f32):
    for i in z:
        if flow.neighbourer_flat.can_leave_domain(i) or flow.neighbourer_flat.nodata(i):
            continue
        iy, ix = flow.neighbourer_flat.rc_from_i(i)
        if ((ix + iy) & 1) != parity:
            continue
        num = 0.0
        den = 0.0
        for k in range(4):
            j = flow.neighbourer_flat.neighbour(i, k)
            if j == -1 or flow.neighbourer_flat.nodata(j):
                continue
            s = ti.abs((z[j] - z[i]) / cte.DX)  # slope proxy
            w = ti.exp(-s / ti.max(1e-6, sigma))
            num += w * (eta[j] - eta[i])
            den += w
        if den > 0:
            eta[i] = eta[i] + tau * num  # den absorbed by tau choice

# ---------------- host helpers (no new fields) ----------------

def _match_volume_inplace(eta, z, acc_vol, cell_area, V_target, iters=24):
    """Find c s.t. sum max(eta+c - z,0)*area == V_target. Bisection on c."""
    volume_shifted(eta, z, 0.0, acc_vol, cell_area)
    V0 = float(acc_vol[None])
    if abs(V0 - V_target) < 1e-12:
        return

    # bracket
    c_lo, c_hi = 0.0, 0.0
    if V0 > V_target:  # need to lower
        step = -max(1.0, 0.5 * (V0 - V_target) / cell_area)
        while True:
            c_lo += step
            volume_shifted(eta, z, c_lo, acc_vol, cell_area)
            if acc_vol[None] <= V_target or abs(c_lo) > 1e6:
                break
    else:  # raise
        step = max(1.0, 0.5 * (V_target - V0) / cell_area)
        while True:
            c_hi += step
            volume_shifted(eta, z, c_hi, acc_vol, cell_area)
            if acc_vol[None] >= V_target or abs(c_hi) > 1e6:
                break

    if c_lo > c_hi:
        c_lo, c_hi = c_hi, c_lo

    # bisection
    for _ in range(iters):
        c_mid = 0.5 * (c_lo + c_hi)
        volume_shifted(eta, z, c_mid, acc_vol, cell_area)
        if acc_vol[None] < V_target:
            c_lo = c_mid
        else:
            c_hi = c_mid

    c_final = 0.5 * (c_lo + c_hi)
    apply_shift_and_project(eta, z, c_final)

# ---------------- main APIs (all fields passed in) ----------------

def projected_diffuse_iter(
    z, h, eta, eta_prev, acc_vol, acc_max,
    cell_area,          # e.g. float(cte.DX*cte.DX)
    tau=0.20,           # <=0.25 safe for 4-neigh
    max_outer=200,
    tol=1e-4,
    use_aniso=False,
    sigma=0.5
):
    """Iterative projected diffusion with exact mass every outer step."""
    init_eta(eta, z, h)
    volume_shifted(eta, z, 0.0, acc_vol, cell_area)
    V_target = float(acc_vol[None])

    copy_field(eta_prev, eta)

    for _ in range(max_outer):
        if use_aniso:
            diffuse_rb_step_aniso(eta, z, 0, tau, sigma)
            diffuse_rb_step_aniso(eta, z, 1, tau, sigma)
        else:
            diffuse_rb_step(eta, z, 0, tau)
            diffuse_rb_step(eta, z, 1, tau)

        project_floor(eta, z)
        _match_volume_inplace(eta, z, acc_vol, cell_area, V_target)

        max_abs_diff(eta, eta_prev, acc_max)
        if float(acc_max[None]) < tol:
            break
        copy_field(eta_prev, eta)

    compute_h_from_eta(h, eta, z)

def one_shot_smooth(
    z, h, eta, acc_vol, acc_max,
    *,
    cell_area,          # e.g. float(cte.DX*cte.DX)
    k_smooth=6,
    tau=0.20,
    use_aniso=False,
    sigma=0.5
):
    """K diffusion sweeps, one projection, exact mass match."""
    init_eta(eta, z, h)
    volume_shifted(eta, z, 0.0, acc_vol, cell_area)
    V_target = float(acc_vol[None])

    for _ in range(k_smooth):
        if use_aniso:
            diffuse_rb_step_aniso(eta, z, 0, tau, sigma)
            diffuse_rb_step_aniso(eta, z, 1, tau, sigma)
        else:
            diffuse_rb_step(eta, z, 0, tau)
            diffuse_rb_step(eta, z, 1, tau)

    project_floor(eta, z)
    _match_volume_inplace(eta, z, acc_vol, cell_area, V_target)
    compute_h_from_eta(h, eta, z)



#######
