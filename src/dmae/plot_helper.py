import numpy as np
import plotly.graph_objects as go


def compare_3d_plots(
    R_iX,
    Q_iX,
    color,
    title,
    axis: bool = False,
    backend: str = "plotly",   # "plotly" or "matplotlib"
    show: bool = True,
):
    """
    Compare 3D truth vs reconstruction.

    Parameters
    ----------
    R_iX : array-like, shape (N, 3)
        Ground-truth / target points.
    Q_iX : array-like, shape (N, 3)
        Reconstructed / predicted points.
    color : array-like, shape (N,)
        Scalar color for reconstruction points.
    title : str
        Figure title.
    axis : bool, default=False
        If False, hide axes, grid, ticks, and background.
        If True, show standard labeled 3D axes.
    backend : {"plotly", "matplotlib"}, default="plotly"
        Visualization backend.
    show : bool, default=True
        Whether to display the figure immediately.

    Returns
    -------
    fig
        Plotly or Matplotlib figure object.
    """
    R_iX = np.asarray(R_iX)
    Q_iX = np.asarray(Q_iX)
    color = np.asarray(color)

    if R_iX.ndim != 2 or R_iX.shape[1] != 3:
        raise ValueError(f"R_iX must have shape (N, 3), got {R_iX.shape}")
    if Q_iX.ndim != 2 or Q_iX.shape[1] != 3:
        raise ValueError(f"Q_iX must have shape (N, 3), got {Q_iX.shape}")
    if R_iX.shape[0] != Q_iX.shape[0]:
        raise ValueError(
            f"R_iX and Q_iX must have the same number of points, "
            f"got {R_iX.shape[0]} and {Q_iX.shape[0]}"
        )
    if color.ndim != 1 or color.shape[0] != Q_iX.shape[0]:
        raise ValueError(
            f"color must have shape (N,), got {color.shape} for N={Q_iX.shape[0]}"
        )

    backend = backend.lower()

    if backend == "plotly":
        fig = go.Figure()

        # Truth: black rings
        fig.add_trace(
            go.Scatter3d(
                x=R_iX[:, 0],
                y=R_iX[:, 1],
                z=R_iX[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color="black",
                    opacity=0.35,
                    symbol="circle-open",
                    line=dict(width=2, color="black"),
                ),
                name="truth",
            )
        )

        # Reconstruction: colored filled markers
        fig.add_trace(
            go.Scatter3d(
                x=Q_iX[:, 0],
                y=Q_iX[:, 1],
                z=Q_iX[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=color,
                    colorscale="Viridis",
                    opacity=0.95,
                    symbol="circle",
                    colorbar=dict(title="color"),
                ),
                name="reconstruction",
            )
        )

        if axis:
            scene = dict(
                xaxis=dict(title="x"),
                yaxis=dict(title="y"),
                zaxis=dict(title="z"),
            )
        else:
            scene = dict(
                xaxis=dict(
                    visible=False,
                    showbackground=False,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    title="",
                ),
                yaxis=dict(
                    visible=False,
                    showbackground=False,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    title="",
                ),
                zaxis=dict(
                    visible=False,
                    showbackground=False,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    title="",
                ),
                bgcolor="rgba(0,0,0,0)",
            )

        fig.update_layout(
            title=title,
            scene=scene,
            width=950,
            height=750,
            margin=dict(l=0, r=0, b=0, t=60),
        )

        if show:
            fig.show()
        return fig

    elif backend == "matplotlib":
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(9.5, 7.5))
        ax = fig.add_subplot(111, projection="3d")

        # Truth: open black circles
        ax.scatter(
            R_iX[:, 0],
            R_iX[:, 1],
            R_iX[:, 2],
            s=40,
            facecolors="none",
            edgecolors="black",
            alpha=0.35,
            linewidths=1.5,
            label="truth",
        )

        # Reconstruction: colored filled points
        sc = ax.scatter(
            Q_iX[:, 0],
            Q_iX[:, 1],
            Q_iX[:, 2],
            c=color,
            s=18,
            alpha=0.95,
            cmap="viridis",
            label="reconstruction",
        )

        ax.set_title(title)

        if axis:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        else:
            ax.set_axis_off()

        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.08)
        cbar.set_label("color")

        # Optional: roughly equal aspect ratio
        mins = np.minimum(R_iX.min(axis=0), Q_iX.min(axis=0))
        maxs = np.maximum(R_iX.max(axis=0), Q_iX.max(axis=0))
        centers = 0.5 * (mins + maxs)
        radius = 0.5 * np.max(maxs - mins)

        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.set_zlim(centers[2] - radius, centers[2] + radius)

        if show:
            plt.show()
        return fig

    else:
        raise ValueError("backend must be either 'plotly' or 'matplotlib'")
