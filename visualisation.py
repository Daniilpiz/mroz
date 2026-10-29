import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_trajectory(f, traj, title, exact=None,
                    pad=0.25, n_grid=100):
    """
    Рисует линии уровня f(x1, x2) и накладывает на них траекторию метода.

    Параметры
    ---------
    f     : Callable[[np.ndarray], float] — функция двух переменных.
    traj  : list[np.ndarray] | np.ndarray — последовательность точек,
            каждая формы (2,). Список массивов тоже принимается.
    title : str — заголовок графика.
    exact : tuple | list | np.ndarray | None — координаты точного
            минимума для отметки на графике.
    pad   : float — доля от диапазона, добавляемая к границам осей.
    n_grid: int — размер сетки для линий уровня.
    """

    # --- 1. Приводим траекторию к ndarray формы (N, 2) ---
    traj = np.asarray(traj, dtype=float)
    if traj.ndim == 1:            # на случай, если передали одну точку
        traj = traj.reshape(1, -1)

    all_pts = traj
    if exact is not None:
        all_pts = np.vstack([all_pts, np.asarray(exact, dtype=float).reshape(1, -1)])

    # --- 2. Авто-пределы по траектории и точному минимуму ---
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()

    if x_max - x_min < 1e-6:
        x_min -= 1.0; x_max += 1.0
    if y_max - y_min < 1e-6:
        y_min -= 1.0; y_max += 1.0

    dx = (x_max - x_min) * pad
    dy = (y_max - y_min) * pad
    xlim = (x_min - dx, x_max + dx)
    ylim = (y_min - dy, y_max + dy)

    # --- 3. Сетка для линий уровня (векторизованно) ---
    xs = np.linspace(xlim[0], xlim[1], n_grid)
    ys = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(lambda a, b: f(np.array([a, b])))(X, Y)

    # --- 4. Уровни: логарифмическая шкала со сдвигом для неположительных значений ---
    zmin, zmax = Z.min(), Z.max()
    if zmax - zmin < 1e-12:
        levels = np.array([zmin])
    else:
        shift = max(0.0, -zmin) + 1e-6
        levels = np.logspace(np.log10(zmin + shift),
                             np.log10(zmax + shift), 30)
        # возвращаем уровни к исходной шкале (без сдвига)
        levels = levels - shift

    # --- 5. Рисуем ---
    fig, ax = plt.subplots(figsize=(8, 7))
    cs = ax.contour(X, Y, Z, levels=levels,
                    cmap='viridis', linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.2g')

    ax.plot(traj[:, 0], traj[:, 1], 'r.-',
            linewidth=1.6, markersize=9, label='траектория')
    ax.plot(traj[0, 0], traj[0, 1], 'go',
            markersize=13, label='старт')
    ax.plot(traj[-1, 0], traj[-1, 1], 'b*',
            markersize=18, label='финиш')

    for i, (xi, yi) in enumerate(traj):
        ax.annotate(str(i), (xi, yi),
                    textcoords="offset points",
                    xytext=(7, 5), fontsize=8, color='darkred')

    if exact is not None:
        ax.plot(exact[0], exact[1], 'kX', markersize=14,
                label='точный минимум')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title, fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_trajectory_3d_plotly(f, traj, title, exact=None,
                              pad=0.25, n_grid=100,
                              xlim=None, ylim=None, zlim=None,
                              colorscale='RdBu_r',
                              surface_opacity=0.9):
        traj = np.asarray(traj, dtype=float)     # ← вот это
        if exact is not None:
            exact = np.asarray(exact, dtype=float)
    # --- 1. Пределы по осям X, Y ---
        all_pts = traj.copy()
        if exact is not None:
            all_pts = np.vstack([all_pts, np.array(exact)])

        if xlim is None:
            x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
            if x_max - x_min < 1e-6:
                x_min -= 1.0; x_max += 1.0
            dx = (x_max - x_min) * pad
            xlim = (x_min - 2*dx, x_max + 2*dx)

        if ylim is None:
            y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
            if y_max - y_min < 1e-6:
                y_min -= 1.0; y_max += 1.0
            dy = (y_max - y_min) * pad
            ylim = (y_min - 2*dy, y_max + 2*dy)

    # --- 2. Сетка поверхности ---
        xs = np.linspace(xlim[0], xlim[1], n_grid)
        ys = np.linspace(ylim[0], ylim[1], n_grid)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    # --- 3. Z-диапазон для вида ---
        Ztraj = np.array([f(p) for p in traj])
        zmin = min(Z.min(), Ztraj.min())
        zmax_raw = max(Z.max(), Ztraj.max())
        zmax_cap = np.percentile(Z, 92)
        zmax = min(zmax_raw, zmax_cap)
        if zlim is not None:
            zmin, zmax = zlim

    # --- 4. Поверхность ---
        fig = go.Figure()

        fig.add_surface(
            x=X, y=Y, z=Z,
            colorscale=colorscale,
            opacity=surface_opacity,
            showscale=True,
            colorbar=dict(title='f(x)', len=0.6),
            contours={
                "z": {"show": True,
                  "usecolormap": False,
                  "highlightcolor": "black",
                  "project": {"z": True}}
            },
            name='поверхность'
        )

    # --- 5. Траектория: белая обводка + красная линия ---
        fig.add_scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=Ztraj,
            mode='lines',
            line=dict(color='white', width=8),
            showlegend=False, hoverinfo='skip'
        )
        fig.add_scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=Ztraj,
            mode='lines+markers+text',
            text=[str(i) for i in range(len(traj))],
            textposition='top center',
            textfont=dict(color='black', size=10),
            line=dict(color='red', width=4),
            marker=dict(size=5, color='red',
                        line=dict(color='white', width=1)),
             name='траектория'
        )

    # --- 6. Вертикальные нити ---
        for i, (xi, yi, zi) in enumerate(zip(traj[:, 0], traj[:, 1], Ztraj)):
            fig.add_scatter3d(
            x=[xi, xi], y=[yi, yi], z=[zmin, zi],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'
        )

    # --- 7. Старт / финиш / точный минимум ---
        fig.add_scatter3d(
        x=[traj[0, 0]], y=[traj[0, 1]], z=[Ztraj[0]],
        mode='markers',
        marker=dict(size=10, color='lime',
                    line=dict(color='black', width=2)),
        name='старт'
    )
        fig.add_scatter3d(
        x=[traj[-1, 0]], y=[traj[-1, 1]], z=[Ztraj[-1]],
        mode='markers',
        marker=dict(size=14, color='blue', symbol='diamond',
                    line=dict(color='white', width=2)),
        name='финиш'
    )
        if exact is not None:
            fig.add_scatter3d(
            x=[exact[0]], y=[exact[1]], z=[f(np.array(exact))],
            mode='markers',
            marker=dict(size=12, color='black', symbol='x'),
            name='точный минимум'
        )

    # --- 8. Оформление ---
        fig.update_layout(
            title=title,
            width=950, height=750,
            scene=dict(
            xaxis=dict(title='x1', range=list(xlim)),
            yaxis=dict(title='x2', range=list(ylim)),
            zaxis=dict(title='f(x)', range=[zmin, zmax]),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0))
        ),
        legend=dict(x=0.02, y=0.98,
                    bgcolor='rgba(255,255,255,0.7)')
    )

        fig.show()