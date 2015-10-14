# Multilinear Reconstruction

## Bilinear Face Model
A bilinear face model can be represented with a third order  tensor $\mathcal C$. This bilinear model can be used to generate new face meshes given identity weights $\mathbf w_{id}$ and expression weights $\mathbf w_{exp}$:
\[\mathcal T = \mathcal C \times_{id} \mathbf w_{id} \times_{exp} \mathbf w_{exp}\]

## Single Image Reconstruction
A set of 2D facial landmarks is generated from bilinear face model $\mathcal C$ with proper identity weights $\mathbf w_{id}$, expression weights $\mathbf w_{exp}$, rigid transformation $\{\mathbf R, \mathbf t\}$ and camera projection matrix $\mathbf M_{proj}$:

\[ \mathbf L = \mathbf M_{proj}\left[\mathbf R \Pi(\mathcal C\times_{id} \mathbf w_{id} \times_{exp} \mathbf w_{exp}) + \mathbf t\right] \]
where $\mathbf L$ is a $n\times2$ matrix of 2D landmarks, and $\Pi$ is a vertex selector for the 3D face mesh.

The goal reconstruction is to recover all unknown parameters ($\mathbf w_{id}$, $\mathbf w_{exp}$, $\mathbf R$, $\mathbf t$, $\mathbf M_{proj}$) from input landmarks.

## Multiple Images Reconstruction
In case multiple images of the same person are available, the reconstruction makes use all images simultaneously to compute best estimation of the unknown parameters. By utilizing the constraint of identity weights, the reconstruction is expected to generate more accurate estimations.
