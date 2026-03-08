import numpy as np
import plotly.graph_objects as go


def compare_3d_plots(
    R_iX,
    Q_iX,
    color,
    title,
    axis: bool = False,
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
            f"R_iX and Q_iX must have the same number of points, got {R_iX.shape[0]} and {Q_iX.shape[0]}"
        )
    if color.ndim != 1 or color.shape[0] != Q_iX.shape[0]:
        raise ValueError(
            f"color must have shape (N,), got {color.shape} for N={Q_iX.shape[0]}"
        )

    fig = go.Figure()

    # Truth: large black rings
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

    # Reconstruction: colored filled discs
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

    fig.show()
    return fig
