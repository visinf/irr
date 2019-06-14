from . import flownet1s
from . import flownet1s_irr
from . import flownet1s_irr_bi
from . import flownet1s_irr_occ
from . import flownet1s_irr_occ_bi
from . import IRR_FlowNet

from . import pwcnet
from . import pwcnet_bi
from . import pwcnet_occ
from . import pwcnet_occ_bi
from . import pwcnet_irr
from . import pwcnet_irr_bi
from . import pwcnet_irr_occ
from . import pwcnet_irr_occ_bi
from . import IRR_PWC


FlowNet1S            = flownet1s.FlowNet1S
FlowNet1S_irr        = flownet1s_irr.FlowNet1S
FlowNet1S_irr_bi     = flownet1s_irr_bi.FlowNet1S
FlowNet1S_irr_occ    = flownet1s_irr_occ.FlowNet1S
FlowNet1S_irr_occ_bi = flownet1s_irr_occ_bi.FlowNet1S

PWCNet               = pwcnet.PWCNet
PWCNet_bi            = pwcnet_bi.PWCNet
PWCNet_occ           = pwcnet_occ.PWCNet
PWCNet_occ_bi        = pwcnet_occ_bi.PWCNet
PWCNet_irr           = pwcnet_irr.PWCNet
PWCNet_irr_bi        = pwcnet_irr_bi.PWCNet
PWCNet_irr_occ       = pwcnet_irr_occ.PWCNet
PWCNet_irr_occ_bi    = pwcnet_irr_occ_bi.PWCNet

IRR_FlowNet          = IRR_FlowNet.FlowNet1S
IRR_PWC              = IRR_PWC.PWCNet

